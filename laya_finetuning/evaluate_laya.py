import json
import argparse
from collections import Counter

import laya
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}")

            rows.append(row)

    return rows


def evaluate(model_dir, dataset_path):
    print(f"Loading model from: {model_dir}")
    agent = laya.load(model_dir)

    rows = load_jsonl(dataset_path)

    y_true = []
    y_pred = []
    mistakes = []

    print(f"Evaluating {len(rows)} examples...\n")

    for i, row in enumerate(rows, 1):
        state = row["state"]
        questions = row["questions"]
        expected = row["expected"]["affiliation_geography"]

        result = agent.predict(state, questions)

        predicted = result["answers"]["affiliation_geography"]["choice"]

        y_true.append(expected)
        y_pred.append(predicted)

        if predicted != expected:
            mistakes.append({
                "state": state,
                "expected": expected,
                "predicted": predicted,
                "raw_result": result,
            })

        if i % 20 == 0 or i == len(rows):
            print(f"Processed {i}/{len(rows)}")

    labels = ["INDIAN", "NOT_INDIAN"]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nPer-class metrics:")
    for label, p, r, f, s in zip(
        labels, precision, recall, f1, support
    ):
        print(
            f"{label:12s} "
            f"Precision={p:.4f} "
            f"Recall={r:.4f} "
            f"F1={f:.4f} "
            f"Support={s}"
        )

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    print("Confusion matrix")
    print("Rows = actual, Columns = predicted")
    print(f"Labels: {labels}")
    print(cm)

    print("\nPrediction distribution:")
    print(Counter(y_pred))

    print(f"\nTotal errors: {len(mistakes)}")

    if mistakes:
        print("\nMisclassified examples:")
        for j, error in enumerate(mistakes, 1):
            print("-" * 70)
            print(f"Error {j}")
            print(f"State:     {error['state']}")
            print(f"Expected:  {error['expected']}")
            print(f"Predicted: {error['predicted']}")

    output_path = dataset_path + ".evaluation.json"

    report = {
        "dataset": dataset_path,
        "model": model_dir,
        "num_examples": len(rows),
        "accuracy": accuracy,
        "labels": labels,
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(
                labels, precision, recall, f1, support
            )
        },
        "confusion_matrix": cm.tolist(),
        "prediction_counts": dict(Counter(y_pred)),
        "errors": mistakes,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved full report to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="./laya_finetuned_SENTRY",
    )

    parser.add_argument(
        "--data",
        required=True,
    )

    args = parser.parse_args()

    evaluate(args.model, args.data)