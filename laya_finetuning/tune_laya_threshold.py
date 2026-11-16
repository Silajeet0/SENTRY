#!/usr/bin/env python3
"""Tune a conservative INDIAN inclusion threshold on a development JSONL set.

Usage:
    python tune_laya_threshold.py --model ./laya_finetuned_SENTRY \
        --data ./data/threshold_dev.jsonl --min-precision 0.99

    # Re-optimize thresholds later without re-running the model:
    python tune_laya_threshold.py --scores ./threshold_results/predictions.csv \
        --min-precision 0.99 --max-fpr 0.01

Do not tune on your final evaluation/test set.
"""
import argparse
import csv
import json
from pathlib import Path

QUESTION = {
    "affiliation_geography": {
        "type": "choice",
        "instructions": (
            "Does this affiliation provide sufficient evidence that the current "
            "institution or campus is located in India?"
        ),
        "criteria": {
            "INDIAN": "Sufficient evidence that the current institution or campus is in India.",
            "NOT_INDIAN": "No sufficient evidence that the current institution or campus is in India.",
        },
    }
}


def normalize_label(value):
    value = str(value).strip()
    if value in ("INDIAN", "positive"):
        return "INDIAN"
    if value in ("NOT_INDIAN", "negative", "ambiguous"):
        return "NOT_INDIAN"
    raise ValueError(f"Unexpected label: {value!r}")


def infer_scores(model_dir, data_path, device):
    import torch
    import laya

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_dir} (device={device})")
    agent = laya.load(str(model_dir), device=device)
    predictions = []
    with open(data_path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            state = row["state"]
            label = normalize_label(row["expected"]["affiliation_geography"])
            answer = agent.predict(state, QUESTION)["answers"]["affiliation_geography"]
            probs = answer.get("probabilities")
            if not isinstance(probs, dict) or "INDIAN" not in probs:
                raise ValueError(f"No INDIAN choice probability at row {number}: {answer}")
            score = float(probs["INDIAN"])
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid probability at row {number}: {score}")
            predictions.append({"row": number, "state": state, "actual": label, "p_indian": score})
            if len(predictions) % 50 == 0:
                print(f"Scored {len(predictions)} rows")
    return predictions


def read_scores(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"actual", "p_indian"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV must contain {sorted(required)} columns")
    out = []
    for row in rows:
        out.append({
            "row": row.get("row", ""),
            "state": row.get("state", ""),
            "actual": normalize_label(row["actual"]),
            "p_indian": float(row["p_indian"]),
        })
    return out


def metrics(rows, threshold):
    tp = fp = tn = fn = 0
    for row in rows:
        pred_positive = row["p_indian"] >= threshold
        actual_positive = row["actual"] == "INDIAN"
        if pred_positive and actual_positive: tp += 1
        elif pred_positive: fp += 1
        elif actual_positive: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
            "accuracy": (tp + tn) / len(rows)}


def report(m):
    print(f"t={m['threshold']:.5f} | Prec={m['precision']:.4f} Rec={m['recall']:.4f} "
          f"FPR={m['fpr']:.4f} F1={m['f1']:.4f} | "
          f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", help="Laya-native development JSONL with state/expected")
    inputs.add_argument("--scores", help="Previously saved predictions.csv, avoids inference")
    parser.add_argument("--model", default="./laya_finetuned_SENTRY")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--min-precision", type=float, default=0.99,
                        help="Minimum measured positive precision on the dev set")
    parser.add_argument("--max-fpr", type=float, default=None,
                        help="Optional maximum measured false-positive rate")
    parser.add_argument("--out", default="./threshold_results")
    args = parser.parse_args()
    for name, value in [("--min-precision", args.min_precision), ("--max-fpr", args.max_fpr)]:
        if value is not None and not 0 <= value <= 1:
            parser.error(f"{name} must be between 0 and 1")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_scores(args.scores) if args.scores else infer_scores(args.model, args.data, args.device)
    if not any(r["actual"] == "INDIAN" for r in rows) or not any(r["actual"] == "NOT_INDIAN" for r in rows):
        parser.error("Need both INDIAN and NOT_INDIAN ground-truth examples for threshold tuning")
    write_csv(out_dir / "predictions.csv", rows)

    # For a >= threshold rule, every distinct score can change the predicted set.
    candidates = sorted({0.5, 1.0, *(r["p_indian"] for r in rows if 0.5 <= r["p_indian"] <= 1.0)})
    all_results = [metrics(rows, t) for t in candidates]
    write_csv(out_dir / "sweep.csv", all_results)
    eligible = [m for m in all_results if m["precision"] >= args.min_precision
                and (args.max_fpr is None or m["fpr"] <= args.max_fpr)]

    print(f"\nN={len(rows)}; INDIAN={sum(r['actual']=='INDIAN' for r in rows)}, "
          f"NOT_INDIAN={sum(r['actual']=='NOT_INDIAN' for r in rows)}")
    print("\nBaseline (argmax-equivalent t=0.5):")
    report(metrics(rows, 0.5))
    print("\nUseful reference thresholds:")
    for threshold in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        report(metrics(rows, threshold))

    if not eligible:
        print("\nNo threshold met the precision/FPR constraints with a positive prediction. "
              "Inspect sweep.csv; collect more diverse validation examples or use a fallback policy.")
        return

    # Maximize recall subject to safety constraint; at equal recall prefer fewer false positives.
    best = max(eligible, key=lambda m: (m["recall"], -m["fp"], m["threshold"]))
    print("\nSelected threshold on DEVELOPMENT data (not final-test performance):")
    report(best)
    (out_dir / "chosen_threshold.json").write_text(
        json.dumps({"selection_rule": "maximize recall subject to precision/FPR constraints",
                    "min_precision": args.min_precision, "max_fpr": args.max_fpr,
                    **best}, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {out_dir / 'predictions.csv'}, sweep.csv, chosen_threshold.json")
    print("Freeze this threshold; evaluate on a separate, untouched test set.")

if __name__ == "__main__":
    main()
