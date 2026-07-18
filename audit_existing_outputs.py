"""
Audit already-extracted indian_papers_structured.json files against the
hardened india_rules.py, WITHOUT re-scraping or re-calling the LLM.

For each Indian-flagged author in each existing output file, re-classify
their raw affiliation string with the new rules. Flags anyone who would
now be "ambiguous" (i.e. was only kept alive by the old bare-acronym
auto-positive rule) so you can review/strip them by hand, or re-run just
that specific paper.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation.india_rules import classify_affiliation

FINAL_OUTPUT_DIR = Path("data/final_output")

def audit():
    suspect_total = 0
    for json_path in sorted(FINAL_OUTPUT_DIR.glob("*/*/indian_papers_structured.json")):
        conference, year = json_path.parent.parent.name, json_path.parent.name
        try:
            papers = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[SKIP] {json_path}: {e}")
            continue

        aff_by_name = {}
        for paper in papers:
            for a in paper.get("all_authors", []):
                if a.get("name"):
                    aff_by_name[a["name"]] = a.get("affiliation", "")

        suspects = []
        for paper in papers:
            for name in paper.get("authors_with_indian_affiliations", []):
                aff = aff_by_name.get(name, "")
                decision = classify_affiliation(aff)
                if decision.label != "positive":
                    suspects.append((paper.get("paper_number"), paper.get("paper_title", "")[:60], name, aff, decision.label))

        if suspects:
            print(f"\n=== {conference} {year}: {len(suspects)} suspect author(s) out of "
                  f"{sum(len(p.get('authors_with_indian_affiliations', [])) for p in papers)} total flagged ===")
            for pnum, ptitle, name, aff, label in suspects:
                print(f"  paper {pnum} ({ptitle}) | {name} | now: {label}")
                print(f"    affiliation: {aff!r}")
            suspect_total += len(suspects)
        else:
            print(f"{conference} {year}: clean ({sum(len(p.get('authors_with_indian_affiliations', [])) for p in papers)} flagged authors, all still positive)")

    print(f"\nTOTAL suspect authors across all conferences: {suspect_total}")

if __name__ == "__main__":
    audit()
