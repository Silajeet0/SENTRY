"""Convert ``benchmark_data.json`` cases to Laya's binary JSONL format.

The independent challenge set uses ``affiliation`` and the original three
labels. The fine-tuned checkpoint uses ``state``, ``questions`` and one binary
``expected`` answer. This parser performs that schema conversion without
adding metadata fields to the Laya input rows.

The binary projection follows the SENTRY fine-tuning policy:

* ``positive`` becomes ``INDIAN``;
* ``negative`` and ``ambiguous`` become ``NOT_INDIAN``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

QUESTION_ID = "affiliation_geography"
SOURCE_LABELS = ("positive", "negative", "ambiguous")
BINARY_LABELS = ("INDIAN", "NOT_INDIAN")
SOURCE_TO_BINARY = {
    "positive": "INDIAN",
    "negative": "NOT_INDIAN",
    "ambiguous": "NOT_INDIAN",
}
QUESTIONS = {
    QUESTION_ID: {
        "type": "choice",
        "instructions": "Does this affiliation provide sufficient evidence that the current institution or campus is located in India?",
        "criteria": {
            "INDIAN": "Sufficient evidence that the current institution or campus is in India.",
            "NOT_INDIAN": "No sufficient evidence that the current institution or campus is in India.",
        },
    }
}


def load_benchmark_data(path: Path) -> dict[str, Any]:
    """Load and validate the independent benchmark source document."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"{path} must contain a top-level cases array")
    seen_ids: set[str] = set()
    for index, case in enumerate(data["cases"], 1):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} is not an object")
        required = {"id", "affiliation", "label", "category", "split"}
        missing = required - set(case)
        if missing:
            raise ValueError(f"case {index} is missing fields: {sorted(missing)}")
        case_id = case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError(f"case {index} has a missing or duplicate id: {case_id!r}")
        seen_ids.add(case_id)
        # An empty affiliation is an intentional missing-evidence challenge
        # case (A041), so preserve it as an empty Laya state.
        if not isinstance(case["affiliation"], str):
            raise ValueError(f"case {case_id} has an invalid affiliation")
        if case["label"] not in SOURCE_LABELS:
            raise ValueError(f"case {case_id} has unsupported label: {case['label']!r}")
        if case["split"] not in {"validation", "test"}:
            raise ValueError(f"case {case_id} has unsupported split: {case['split']!r}")
    return data


def project_case(case: dict[str, Any]) -> dict[str, Any]:
    """Convert one source case to the exact Laya JSONL row schema."""
    return {
        "state": case["affiliation"],
        "questions": QUESTIONS,
        "expected": {QUESTION_ID: SOURCE_TO_BINARY[case["label"]]},
    }


def parse_cases(data: dict[str, Any], split: str = "all") -> list[dict[str, Any]]:
    """Return converted Laya rows, optionally restricted to a source split."""
    if split not in {"all", "validation", "test"}:
        raise ValueError(f"unsupported split: {split!r}")
    return [project_case(case) for case in data["cases"] if split == "all" or case["split"] == split]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("cannot write an empty Laya dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=root / "benchmark/benchmark_data.json")
    parser.add_argument("--output", type=Path, default=root / "benchmark/results/benchmark_data_laya.jsonl")
    parser.add_argument("--split", choices=("all", "validation", "test"), default="all")
    args = parser.parse_args(argv)
    data = load_benchmark_data(args.input)
    rows = parse_cases(data, args.split)
    write_jsonl(rows, args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "split": args.split, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
