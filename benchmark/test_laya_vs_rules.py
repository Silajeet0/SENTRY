"""Benchmark the fine-tuned Laya checkpoint against SENTRY's rule classifier.

The two systems are evaluated on the same binary fine-tuning test split. The
fine-tuned model is expected to return ``INDIAN`` or ``NOT_INDIAN``. The existing
deterministic classifier returns ``positive``, ``negative`` or ``ambiguous``;
for this binary benchmark, ``positive`` maps to ``INDIAN`` and the other two
labels map to ``NOT_INDIAN``. The raw rule label is retained in the report.

"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.india_rules import classify_affiliation
from evaluation.laya_classifier import DEFAULT_LAYA_THRESHOLD

QUESTION_ID = "affiliation_geography"
LABELS = ("INDIAN", "NOT_INDIAN")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_test_data(path: Path) -> list[dict[str, Any]]:
    """Load and validate the requested ``state/questions/expected`` JSONL."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        if set(row) != {"state", "questions", "expected"}:
            raise ValueError(
                f"{path}:{line_number} must contain exactly state, questions and expected; "
                f"found {sorted(row)}"
            )
        if not isinstance(row["state"], str) or not row["state"].strip():
            raise ValueError(f"{path}:{line_number} has an invalid state")
        questions = row["questions"]
        expected = row["expected"]
        if not isinstance(questions, dict) or set(questions) != {QUESTION_ID}:
            raise ValueError(f"{path}:{line_number} must contain question {QUESTION_ID!r}")
        if not isinstance(expected, dict) or set(expected) != {QUESTION_ID}:
            raise ValueError(f"{path}:{line_number} must contain expected[{QUESTION_ID!r}]")
        label = expected[QUESTION_ID]
        if label not in LABELS:
            raise ValueError(f"{path}:{line_number} has unsupported expected label {label!r}")
        question = questions[QUESTION_ID]
        if not isinstance(question, dict) or question.get("type") != "choice":
            raise ValueError(f"{path}:{line_number} question is not a choice question")
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or set(criteria) != set(LABELS):
            raise ValueError(f"{path}:{line_number} criteria must contain {LABELS}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no examples")
    return rows


def binary_rule_label(raw_label: str) -> str:
    """Apply SENTRY's binary policy to a deterministic classifier result."""
    if raw_label == "positive":
        return "INDIAN"
    if raw_label in {"negative", "ambiguous"}:
        return "NOT_INDIAN"
    raise ValueError(f"Unknown deterministic classifier label: {raw_label!r}")


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = {truth: {prediction: 0 for prediction in LABELS} for truth in LABELS}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    total = len(rows)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    recalls = []
    for label in LABELS:
        true_positive = matrix[label][label]
        predicted = sum(matrix[truth][label] for truth in LABELS)
        support = sum(matrix[label].values())
        precision = _safe_rate(true_positive, predicted)
        recall = _safe_rate(true_positive, support)
        f1 = _safe_rate(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
        recalls.append(recall)
    accuracy = _safe_rate(sum(matrix[label][label] for label in LABELS), total)
    return {
        "n": total,
        "accuracy": accuracy,
        "macro_f1": statistics.mean(f1_values) if f1_values else 0.0,
        "balanced_accuracy": statistics.mean(recalls) if recalls else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _latency_metrics(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "mean_ms": statistics.mean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
    }


class FineTunedLaya:
    def __init__(
        self,
        model_path: Path,
        device: str,
        threads: int,
        threshold: float = DEFAULT_LAYA_THRESHOLD,
    ) -> None:
        os.environ.setdefault("HF_HOME", str(ROOT / "data" / "model_cache"))
        os.environ.setdefault("USE_TF", "0")
        import torch
        import laya

        torch.set_num_threads(threads)
        torch.manual_seed(0)
        self.model_path = model_path
        self.device = device
        self.threads = threads
        self.threshold = threshold
        self.agent = laya.load(str(model_path), device=device)
        self.metadata = {
            "model": str(model_path),
            "device": device,
            "threads": threads,
            "threshold": threshold,
            "config_sha256": _sha256(model_path / "rl_agent_config.json")
            if (model_path / "rl_agent_config.json").exists()
            else None,
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("laya", "torch", "transformers", "huggingface-hub")
            },
        }

    def predict(self, state: str, questions: dict[str, Any]) -> tuple[str, dict[str, float], dict[str, Any]]:
        response = self.agent.predict("Current affiliation: " + state, questions)
        try:
            answer = response["answers"][QUESTION_ID]
            choice = answer["choice"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Laya response has no {QUESTION_ID} choice: {response!r}") from exc
        if choice not in LABELS:
            raise ValueError(f"Fine-tuned Laya returned unsupported label {choice!r}")
        probabilities = answer.get("probabilities") or {}
        if set(probabilities) != set(LABELS):
            raise ValueError(f"Laya probabilities do not contain exactly {LABELS}: {probabilities!r}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in probabilities.values()):
            raise ValueError(f"Laya returned invalid probabilities: {probabilities!r}")
        total = sum(probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=0.002):
            raise ValueError(f"Laya probabilities do not sum to one: {probabilities!r}")
        normalized = {label: probabilities[label] / total for label in LABELS}
        raw_label = max(LABELS, key=normalized.__getitem__)
        predicted = "INDIAN" if normalized["INDIAN"] >= self.threshold else "NOT_INDIAN"
        return predicted, normalized, response


def evaluate_rules(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = []
    timings = []
    for row in rows:
        started = time.perf_counter()
        decision = classify_affiliation(row["state"])
        timings.append((time.perf_counter() - started) * 1000)
        predictions.append(
            {
                "state": row["state"],
                "expected": row["expected"][QUESTION_ID],
                "raw_prediction": decision.label,
                "predicted": binary_rule_label(decision.label),
                "matches": decision.matches,
            }
        )
    return {"metrics": binary_metrics(predictions), "latency": _latency_metrics(timings), "predictions": predictions}


def evaluate_laya(rows: list[dict[str, Any]], classifier: FineTunedLaya) -> dict[str, Any]:
    predictions = []
    timings = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        choice, probabilities, response = classifier.predict(row["state"], row["questions"])
        timings.append((time.perf_counter() - started) * 1000)
        predictions.append(
            {
                "state": row["state"],
                "expected": row["expected"][QUESTION_ID],
                "predicted": choice,
                "probabilities": probabilities,
                "confidence": response["answers"][QUESTION_ID].get("confidence"),
            }
        )
        if index % 10 == 0:
            print(f"laya: {index}/{len(rows)}", file=sys.stderr, flush=True)
    return {"metrics": binary_metrics(predictions), "latency": _latency_metrics(timings), "predictions": predictions}


def build_disagreements(rule_result: dict[str, Any], laya_result: dict[str, Any]) -> list[dict[str, Any]]:
    disagreements = []
    for rule, laya in zip(rule_result["predictions"], laya_result["predictions"]):
        if rule["predicted"] != laya["predicted"]:
            disagreements.append(
                {
                    "state": rule["state"],
                    "expected": rule["expected"],
                    "rule_prediction": rule["predicted"],
                    "rule_raw_prediction": rule["raw_prediction"],
                    "rule_matches": rule["matches"],
                    "laya_prediction": laya["predicted"],
                    "laya_probabilities": laya["probabilities"],
                }
            )
    return disagreements


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "laya_finetuning/finetuning_data/test.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "laya_finetuning/laya_finetuned_SENTRY")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark/results/laya_vs_rules_test_report.json")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-cases", type=int, help="Evaluate only the first N cases for a smoke run")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if not args.data.is_file():
        parser.error(f"Test data does not exist: {args.data}")
    if not args.model.is_dir():
        parser.error(f"Model directory does not exist: {args.model}")
    rows = load_test_data(args.data)
    if args.max_cases is not None:
        if args.max_cases < 1:
            parser.error("--max-cases must be at least 1")
        rows = rows[: args.max_cases]
    started = time.perf_counter()
    classifier = FineTunedLaya(args.model, args.device, args.threads)
    load_seconds = time.perf_counter() - started
    rule_result = evaluate_rules(rows)
    laya_result = evaluate_laya(rows, classifier)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "data_sha256": _sha256(args.data),
        "model": classifier.metadata,
        "model_load_seconds": load_seconds,
        "n": len(rows),
        "label_counts": dict(Counter(row["expected"][QUESTION_ID] for row in rows)),
        "binary_rule_mapping": {
            "positive": "INDIAN",
            "negative": "NOT_INDIAN",
            "ambiguous": "NOT_INDIAN",
        },
        "results": {"rules": rule_result, "laya": laya_result},
        "disagreements": build_disagreements(rule_result, laya_result),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({name: result["metrics"] for name, result in report["results"].items()}, indent=2))
    print(f"Disagreements: {len(report['disagreements'])}/{len(rows)}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
