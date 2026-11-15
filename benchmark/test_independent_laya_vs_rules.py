"""Evaluate deterministic rules and fine-tuned Laya on benchmark_data.json.

This is an independent evaluation from the fine-tuning corpus. The source
benchmark has positive/negative/ambiguous labels; the parser projects them to
the binary policy used by the checkpoint and writes Laya-compatible JSONL before
inference.

Run from the repository root:

    .venv/bin/python benchmark/test_independent_laya_vs_rules.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.india_rules import classify_affiliation
from benchmark.parse_benchmark_data import (
    BINARY_LABELS,
    QUESTION_ID,
    SOURCE_TO_BINARY,
    load_benchmark_data,
    parse_cases,
    write_jsonl,
)
from benchmark.test_laya_vs_rules import (
    FineTunedLaya,
    _latency_metrics,
    _sha256,
    binary_metrics,
    binary_rule_label,
)


def plot_metrics(report: dict[str, Any], plot_dir: Path) -> list[str]:
    """Save comparison figures and return their repository-relative paths."""
    # Use a non-interactive backend so this works on servers and CI runners.
    mpl_config = ROOT / "data" / "matplotlib"
    mpl_cache = ROOT / "data" / "cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    split = report["split"]
    prefix = f"independent_{split}"
    rules = report["results"]["rules"]
    laya = report["results"]["laya"]
    saved: list[str] = []

    def save(fig: Any, name: str) -> None:
        path = plot_dir / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        try:
            saved.append(str(path.relative_to(ROOT)))
        except ValueError:
            saved.append(str(path))

    # Overall statistical comparison.
    metric_names = ("accuracy", "macro_f1", "balanced_accuracy")
    metric_titles = ("Accuracy", "Macro-F1", "Balanced accuracy")
    x = list(range(len(metric_names)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    rule_values = [rules["metrics"][metric] for metric in metric_names]
    laya_values = [laya["metrics"][metric] for metric in metric_names]
    bars_rules = ax.bar([value - width / 2 for value in x], rule_values, width, label="Rules", color="#4c78a8")
    bars_laya = ax.bar([value + width / 2 for value in x], laya_values, width, label="Fine-tuned Laya", color="#f58518")
    ax.set_xticks(x, metric_titles)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Independent benchmark: overall metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for bars in (bars_rules, bars_laya):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    save(fig, "overall_metrics")

    # Precision, recall and F1 for both binary classes.
    class_names = list(BINARY_LABELS)
    series = [
        ("Rules · INDIAN", rules["metrics"]["per_class"][class_names[0]]),
        ("Rules · NOT_INDIAN", rules["metrics"]["per_class"][class_names[1]]),
        ("Laya · INDIAN", laya["metrics"]["per_class"][class_names[0]]),
        ("Laya · NOT_INDIAN", laya["metrics"]["per_class"][class_names[1]]),
    ]
    metric_names = ("precision", "recall", "f1")
    metric_titles = ("Precision", "Recall", "F1")
    x = list(range(len(metric_names)))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ("#4c78a8", "#72b7b2", "#f58518", "#e45756")
    for index, ((series_name, values), color) in enumerate(zip(series, colors)):
        offsets = [value + (index - 1.5) * width for value in x]
        bars = ax.bar(offsets, [values[metric] for metric in metric_names], width, label=series_name, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(x, metric_titles)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Independent benchmark: per-class metrics")
    ax.legend(fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "per_class_metrics")

    # Raw-count confusion matrices make the error direction explicit.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    maximum = max(
        max(max(row.values()) for row in rules["metrics"]["confusion_matrix"].values()),
        max(max(row.values()) for row in laya["metrics"]["confusion_matrix"].values()),
        1,
    )
    for ax, title, result in zip(axes, ("Rules", "Fine-tuned Laya"), (rules, laya)):
        matrix = result["metrics"]["confusion_matrix"]
        values = [[matrix[truth][prediction] for prediction in BINARY_LABELS] for truth in BINARY_LABELS]
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
        ax.set_xticks(range(2), BINARY_LABELS)
        ax.set_yticks(range(2), BINARY_LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        for row_index in range(2):
            for column_index in range(2):
                ax.text(column_index, row_index, str(values[row_index][column_index]), ha="center", va="center")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Cases")
    save(fig, "confusion_matrices")

    # Category-level accuracy highlights where the two systems differ.
    categories = sorted(set(rules["by_category"]) | set(laya["by_category"]))
    if categories:
        y = list(range(len(categories)))
        fig_height = max(6, 0.32 * len(categories) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        rule_values = [rules["by_category"].get(category, {}).get("accuracy", 0.0) for category in categories]
        laya_values = [laya["by_category"].get(category, {}).get("accuracy", 0.0) for category in categories]
        height = 0.36
        ax.barh([value - height / 2 for value in y], rule_values, height, label="Rules", color="#4c78a8")
        ax.barh([value + height / 2 for value in y], laya_values, height, label="Fine-tuned Laya", color="#f58518")
        ax.set_yticks(y, categories)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Accuracy")
        ax.set_title("Independent benchmark: accuracy by challenge category")
        ax.legend()
        ax.grid(axis="x", alpha=0.25)
        save(fig, "category_accuracy")

    return saved


def evaluate_rules(cases: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = []
    timings = []
    for case in cases:
        started = time.perf_counter()
        decision = classify_affiliation(case["affiliation"])
        timings.append((time.perf_counter() - started) * 1000)
        predictions.append(
            {
                "id": case["id"],
                "category": case["category"],
                "split": case["split"],
                "source_label": case["label"],
                "expected": SOURCE_TO_BINARY[case["label"]],
                "raw_prediction": decision.label,
                "predicted": binary_rule_label(decision.label),
                "matches": decision.matches,
                "state": case["affiliation"],
            }
        )
    return {"metrics": binary_metrics(predictions), "latency": _latency_metrics(timings), "predictions": predictions}


def evaluate_laya(cases: list[dict[str, Any]], laya_rows: list[dict[str, Any]], classifier: FineTunedLaya) -> dict[str, Any]:
    predictions = []
    timings = []
    for index, (case, laya_row) in enumerate(zip(cases, laya_rows), 1):
        started = time.perf_counter()
        choice, probabilities, response = classifier.predict(laya_row["state"], laya_row["questions"])
        timings.append((time.perf_counter() - started) * 1000)
        predictions.append(
            {
                "id": case["id"],
                "category": case["category"],
                "split": case["split"],
                "source_label": case["label"],
                "expected": SOURCE_TO_BINARY[case["label"]],
                "predicted": choice,
                "probabilities": probabilities,
                "confidence": response["answers"][QUESTION_ID].get("confidence"),
                "state": case["affiliation"],
            }
        )
        if index % 10 == 0:
            print(f"laya: {index}/{len(cases)}", file=sys.stderr, flush=True)
    return {"metrics": binary_metrics(predictions), "latency": _latency_metrics(timings), "predictions": predictions}


def grouped_metrics(result: dict[str, Any], field: str) -> dict[str, Any]:
    groups = sorted({row[field] for row in result["predictions"]})
    return {
        group: binary_metrics([row for row in result["predictions"] if row[field] == group])
        for group in groups
    }


def disagreements(rule_result: dict[str, Any], laya_result: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for rule, laya in zip(rule_result["predictions"], laya_result["predictions"]):
        if rule["predicted"] != laya["predicted"]:
            output.append(
                {
                    "id": rule["id"],
                    "state": rule["state"],
                    "category": rule["category"],
                    "source_label": rule["source_label"],
                    "expected": rule["expected"],
                    "rule_prediction": rule["predicted"],
                    "rule_raw_prediction": rule["raw_prediction"],
                    "rule_matches": rule["matches"],
                    "laya_prediction": laya["predicted"],
                    "laya_probabilities": laya["probabilities"],
                }
            )
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "benchmark/benchmark_data.json")
    parser.add_argument("--parsed-output", type=Path, default=ROOT / "benchmark/results/benchmark_data_laya.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark/results/independent_laya_vs_rules_report.json")
    parser.add_argument("--plot-dir", type=Path, default=ROOT / "benchmark/results", help="Directory for PNG metric plots")
    parser.add_argument("--model", type=Path, default=ROOT / "laya_finetuning/laya_finetuned_SENTRY")
    parser.add_argument("--split", choices=("all", "validation", "test"), default="all")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-cases", type=int, help="Evaluate only the first N cases for a smoke run")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if not args.input.is_file():
        parser.error(f"Input benchmark does not exist: {args.input}")
    if not args.model.is_dir():
        parser.error(f"Model directory does not exist: {args.model}")

    source = load_benchmark_data(args.input)
    cases = [case for case in source["cases"] if args.split == "all" or case["split"] == args.split]
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("--max-cases must be at least 1")
        cases = cases[: args.max_cases]
    if not cases:
        parser.error("Selected split contains no cases")
    selected_source = {**source, "cases": cases}
    laya_rows = parse_cases(selected_source, "all")
    write_jsonl(laya_rows, args.parsed_output)

    started = time.perf_counter()
    classifier = FineTunedLaya(args.model, args.device, args.threads)
    load_seconds = time.perf_counter() - started
    rule_result = evaluate_rules(cases)
    laya_result = evaluate_laya(cases, laya_rows, classifier)
    disagreement_rows = disagreements(rule_result, laya_result)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "parsed_output": str(args.parsed_output),
        "parsed_output_sha256": _sha256(args.parsed_output),
        "source_dataset_version": source.get("dataset_version"),
        "split": args.split,
        "n": len(cases),
        "source_label_counts": dict(Counter(case["label"] for case in cases)),
        "binary_expected_counts": dict(Counter(SOURCE_TO_BINARY[case["label"]] for case in cases)),
        "source_to_binary": SOURCE_TO_BINARY,
        "model": classifier.metadata,
        "model_load_seconds": load_seconds,
        "results": {
            "rules": {
                **rule_result,
                "by_source_label": grouped_metrics(rule_result, "source_label"),
                "by_category": grouped_metrics(rule_result, "category"),
            },
            "laya": {
                **laya_result,
                "by_source_label": grouped_metrics(laya_result, "source_label"),
                "by_category": grouped_metrics(laya_result, "category"),
            },
        },
        "disagreements": disagreement_rows,
        "environment": {"python": sys.version, "binary_labels": BINARY_LABELS},
    }
    report["plots"] = plot_metrics(report, args.plot_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({name: result["metrics"] for name, result in report["results"].items()}, indent=2))
    print(f"Disagreements: {len(disagreement_rows)}/{len(cases)}")
    print(f"Parsed Laya data: {args.parsed_output}")
    print("Plots:")
    for plot in report["plots"]:
        print(f"  {plot}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
