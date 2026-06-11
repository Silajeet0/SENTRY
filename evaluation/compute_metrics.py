"""
Compute retrieval metrics for Indian-affiliated paper detection.

Ground truth is produced by evaluation/ground_truth_generator.py.
Predictions are the pipeline's indian_papers_structured.json, which stores only
predicted-positive papers.
"""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse


def document_key(url: str) -> str:
    match = re.search(r"ieeexplore\.ieee\.org/document/(\d+)", url)
    if match:
        return match.group(1)

    path_match = re.search(r"/document/(\d+)", urlparse(url).path)
    if path_match:
        return path_match.group(1)

    return url.rstrip("/")


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def load_json(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_metrics(ground_truth: list[dict], predictions: list[dict], include_ambiguous: bool) -> dict:
    prediction_ids = {document_key(record["paper_url"]) for record in predictions}

    evaluated = []
    skipped = []
    for record in ground_truth:
        label = record.get("label", "unknown")
        if label == "unknown" or (label == "ambiguous" and not include_ambiguous):
            skipped.append(record)
            continue
        evaluated.append(record)

    tp = fp = tn = fn = 0
    false_positives = []
    false_negatives = []

    for record in evaluated:
        key = document_key(record["paper_url"])
        truth_positive = record["label"] in {"positive", "ambiguous"}
        predicted_positive = key in prediction_ids

        if truth_positive and predicted_positive:
            tp += 1
        elif truth_positive and not predicted_positive:
            fn += 1
            false_negatives.append(record)
        elif not truth_positive and predicted_positive:
            fp += 1
            false_positives.append(record)
        else:
            tn += 1

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    false_positive_rate = safe_divide(fp, fp + tn)
    false_negative_rate = safe_divide(fn, fn + tp)
    balanced_accuracy = (recall + specificity) / 2 if evaluated else 0.0

    return {
        "counts": {
            "evaluated": len(evaluated),
            "skipped": len(skipped),
            "ground_truth_positive": sum(1 for r in evaluated if r["label"] in {"positive", "ambiguous"}),
            "ground_truth_negative": sum(1 for r in evaluated if r["label"] == "negative"),
            "predicted_positive": len(prediction_ids),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "skipped_records": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute AEGIS Indian-affiliation retrieval metrics.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground_truth.json.")
    parser.add_argument("--predictions", required=True, help="Path to indian_papers_structured.json.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path.")
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="Count ambiguous ground-truth labels as positives. Default: skip them.",
    )
    args = parser.parse_args()

    ground_truth = load_json(args.ground_truth)
    predictions = load_json(args.predictions)
    result = compute_metrics(ground_truth, predictions, args.include_ambiguous)

    print(json.dumps({
        "counts": result["counts"],
        "metrics": result["metrics"],
    }, indent=2))

    output = Path(args.output) if args.output else Path(args.predictions).parent / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved full metrics report to: {output}")


if __name__ == "__main__":
    main()
