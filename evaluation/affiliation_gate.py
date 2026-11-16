"""Re-verify extracted submission candidates with the production Laya model."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.laya_classifier import classify_affiliation


ROOT = Path(__file__).resolve().parents[1]
FINAL_OUTPUT_DIR = ROOT / "data" / "final_output"


def verify_papers(papers: list[dict], conference: str, year: str) -> dict:
    """Return verified papers and persist any Laya disagreements for review."""
    verified = []
    flagged = []
    for paper in papers:
        authors = {a.get("name"): a.get("affiliation", "") for a in paper.get("all_authors", []) if a.get("name")}
        reasons = []
        for name in paper.get("authors_with_indian_affiliations", []):
            affiliation = authors.get(name, "")
            if not affiliation:
                reasons.append({"author": name, "reason": "missing affiliation in extracted author record"})
                continue
            try:
                decision = classify_affiliation(affiliation)
            except Exception as exc:
                reasons.append({"author": name, "reason": f"Laya classification failed: {exc}"})
                continue
            if decision.label != "INDIAN":
                reasons.append({
                    "author": name,
                    "affiliation": affiliation,
                    "reason": "Laya returned NOT_INDIAN",
                    "probabilities": decision.probabilities,
                })
        if reasons:
            flagged.append({"paper": paper, "reasons": reasons})
        else:
            verified.append(paper)

    output_dir = FINAL_OUTPUT_DIR / str(conference) / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "need_human_review.json"
    review_path.write_text(json.dumps(flagged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "total_papers": len(papers),
        "verified_count": len(verified),
        "flagged_count": len(flagged),
        "flagged_papers": flagged,
        "verified_papers": verified,
        "need_human_review_path": str(review_path),
    }


def run_gate(conference: str, year: str) -> dict:
    path = FINAL_OUTPUT_DIR / str(conference) / str(year) / "indian_papers_structured.json"
    if not path.exists():
        raise FileNotFoundError(f"No extracted paper data found at {path}")
    papers = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(papers, list):
        raise ValueError(f"Expected a JSON array at {path}")
    return verify_papers(papers, conference, year)
