"""Evaluate Laya against fixed labels and the current SENTRY rule baseline.

Run: python benchmark/test_laya.py --backend both --model convaiinnovations/laya
"""
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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.india_rules import classify_affiliation

LABELS = ("positive", "negative", "ambiguous")
QUESTIONS = {"affiliation": {
    "type": "choice",
    "instructions": "Classify current institutional affiliation by campus location. Ignore author nationality, emails, former jobs and India as a research subject. Bare colliding acronyms or missing locations are ambiguous. Overseas campuses are non-Indian. At least one current Indian affiliation makes the result positive.",
    "criteria": {
        "positive": "At least one current institutional affiliation is located in India.",
        "negative": "Current institutional affiliations are identified and all are outside India.",
        "ambiguous": "Insufficient evidence to determine current institution or location, and no established Indian affiliation."
    }
}}


def combine(labels):
    return "positive" if "positive" in labels else "ambiguous" if "ambiguous" in labels else "negative"


def load_dataset(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data["cases"]
    ids = set()
    if not cases:
        raise ValueError("Benchmark has no cases")
    for case in cases:
        if case["id"] in ids or case["label"] not in LABELS or not isinstance(case["affiliation"], str):
            raise ValueError(f"Invalid or duplicate case: {case['id']}")
        if case.get("split") not in {"validation", "test"}:
            raise ValueError(f"Missing split: {case['id']}")
        ids.add(case["id"])
    labels = {c["id"]: c["label"] for c in cases}
    for paper in data.get("paper_cases", []):
        if not paper["case_ids"] or not set(paper["case_ids"]) <= ids:
            raise ValueError(f"Invalid paper fixture: {paper['id']}")
        if combine([labels[i] for i in paper["case_ids"]]) != paper["label"]:
            raise ValueError(f"Inconsistent paper label: {paper['id']}")
    return data


def validate_probabilities(answer):
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(LABELS):
        raise ValueError("Laya response lacks probabilities for the three requested labels")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
        raise ValueError("Invalid Laya probabilities")
    total = sum(probs.values())
    if abs(total-1) > 0.002:
        raise ValueError("Laya probabilities do not sum to one within rounding tolerance")
    if answer.get("choice") not in LABELS:
        raise ValueError("Laya returned an unknown label")
    return {k: v / total for k,v in probs.items()}


class LayaClassifier:
    def __init__(self, model, revision=None, device="cpu", threads=2):
        # Keep downloads within the project rather than writing to the user's home.
        os.environ.setdefault("HF_HOME", str(ROOT / "data" / "model_cache"))
        os.environ.setdefault("USE_TF", "0")
        import torch
        import laya
        from huggingface_hub import snapshot_download
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        if Path(model).is_dir():
            location = model
            resolved = "local:" + hashlib.sha256((Path(model) / "rl_agent_config.json").read_bytes()).hexdigest()
        else:
            location = snapshot_download(model, revision=revision,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"], ignore_patterns=["*/*"])
            resolved = Path(location).name
        self.agent = laya.load(location, device=device)
        self.metadata = {"model": model, "revision": resolved, "device": device, "threads": threads,
                         "versions": {p: importlib.metadata.version(p) for p in ("laya", "torch", "transformers", "huggingface-hub")}}

    def predict(self, affiliation):
        response = self.agent.predict("Current affiliation: " + affiliation, QUESTIONS)
        answer = response["answers"]["affiliation"]
        return answer["choice"], validate_probabilities(answer)


def metrics(rows):
    matrix = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
    for row in rows:
        matrix[row["expected"]][row["predicted"]] += 1
    n = len(rows)
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        predicted = sum(matrix[t][label] for t in LABELS)
        support = sum(matrix[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0, "support": support}
    accuracy = sum(matrix[l][l] for l in LABELS)/n if n else 0.0
    result = {"n": n, "accuracy": accuracy, "macro_f1": sum(v["f1"] for v in per_class.values())/3,
              "per_class": per_class, "confusion_matrix": matrix,
              "pipeline_inclusion": {"true_positive": matrix["positive"]["positive"],
                  "false_positive": sum(matrix[l]["positive"] for l in ("negative", "ambiguous")),
                  "false_negative": sum(matrix["positive"][l] for l in ("negative", "ambiguous")),
                  "true_negative": sum(matrix[t][p] for t in ("negative", "ambiguous") for p in ("negative", "ambiguous"))}}
    probability_rows = [r for r in rows if r.get("probabilities")]
    if probability_rows:
        brier = sum(sum((r["probabilities"][l] - (r["expected"] == l))**2 for l in LABELS) for r in probability_rows)/len(probability_rows)
        ece = 0.0
        for b in range(10):
            group = [r for r in probability_rows if min(int(max(r["probabilities"].values())*10),9) == b]
            if group:
                confidence = statistics.mean(max(r["probabilities"].values()) for r in group)
                correct = statistics.mean(r["predicted"] == r["expected"] for r in group)
                ece += len(group)/len(probability_rows)*abs(confidence-correct)
        result["calibration"] = {"multiclass_brier_sum": brier, "ece_10_equal_width_bins": ece,
                                 "note": "Descriptive only; confidence is not calibrated for affiliation geography."}
    if rows and "latency_ms" in rows[0]:
        times = sorted(r["latency_ms"] for r in rows)
        result["latency_ms"] = {"mean": statistics.mean(times), "p50": statistics.median(times), "p95": times[max(0, math.ceil(.95*len(times))-1)]}
    return result


def evaluate(cases, classifier, backend, paper_cases):
    rows = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        label, probabilities = classifier(case["affiliation"])
        rows.append({"id": case["id"], "affiliation": case["affiliation"], "category": case["category"],
                     "split": case["split"], "expected": case["label"], "predicted": label,
                     "probabilities": probabilities, "latency_ms": (time.perf_counter()-started)*1000})
        if index % 10 == 0:
            print(f"{backend}: {index}/{len(cases)}", file=sys.stderr, flush=True)
    by_id = {r["id"]: r for r in rows}
    papers = [{"id": p["id"], "expected": p["label"], "predicted": combine([by_id[i]["predicted"] for i in p["case_ids"]])}
              for p in paper_cases if set(p["case_ids"]) <= by_id.keys()]
    return {"overall": metrics(rows), "by_split": {s: metrics([r for r in rows if r["split"] == s]) for s in sorted({r["split"] for r in rows})},
            "by_category": {s: metrics([r for r in rows if r["category"] == s]) for s in sorted({r["category"] for r in rows})},
            "paper_level": {"metrics": metrics(papers), "predictions": papers,
                "scope": "Replay of structured affiliation decisions and positive-only inclusion, not scraping or LLM extraction."},
            "errors": [r for r in rows if r["expected"] != r["predicted"]], "predictions": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmark/benchmark_data.json")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark/results/laya_report.json")
    parser.add_argument("--backend", choices=["rules", "laya", "both"], default="both")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--revision", help="Hugging Face commit hash for exact reproduction")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--split", choices=["all", "validation", "test"], default="all")
    args = parser.parse_args(argv)
    data = load_dataset(args.dataset)
    cases = [c for c in data["cases"] if args.split == "all" or c["split"] == args.split]
    if not cases:
        parser.error("Selected split is empty")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset_version": data["dataset_version"],
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(), "python": platform.python_version(),
              "platform": platform.platform(), "policy": data["policy"], "questions": QUESTIONS,
              "split": args.split, "status": "running", "results": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        temporary.replace(args.output)
    save()
    try:
        if args.backend in {"rules", "both"}:
            report["results"]["rules"] = evaluate(cases, lambda a: (classify_affiliation(a).label, None), "rules", data.get("paper_cases", []))
            save()
        if args.backend in {"laya", "both"}:
            started = time.perf_counter()
            classifier = LayaClassifier(args.model, args.revision, args.device, args.threads)
            report["laya_runtime"] = {**classifier.metadata, "load_seconds": time.perf_counter()-started}
            report["results"]["laya"] = evaluate(cases, classifier.predict, "laya", data.get("paper_cases", []))
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    save()
    print(json.dumps({name: value["overall"] for name,value in report["results"].items()}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
