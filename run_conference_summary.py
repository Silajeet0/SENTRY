"""
Run the email-summary feature for one already-extracted conference/year
directly (no orchestrator/agent involved) — for debugging/smoke-testing,
the same role run_single_paper.py plays for the main extraction pipeline.

Requires data/final_output/<conference>/<year>/indian_papers_structured.json
to already exist (i.e. run_pipeline has already completed for it).

Usage:
    python run_conference_summary.py ICML 2025
    python run_conference_summary.py ICML 2025 --refresh-cache
    python run_conference_summary.py ICML 2025 --max-papers 3   # quick smoke test
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from summarizer.abstract_fetcher import fetch_abstracts_for_papers
from summarizer.email_summarizer import SummaryLLM, build_email


def main():
    parser = argparse.ArgumentParser(description="Generate an Indian-authors email summary for one conference/year.")
    parser.add_argument("conference", help="e.g. 'ICML'")
    parser.add_argument("year", help="e.g. '2025'")
    parser.add_argument(
        "--refresh-cache", action="store_true",
        help="Ignore any cached abstracts from a previous run and re-fetch all of them.",
    )
    parser.add_argument(
        "--max-papers", type=int, default=None,
        help="Only process the first N papers — useful for a quick smoke test before a full run.",
    )
    parser.add_argument(
        "--delay", type=float, default=3,
        help="Seconds to wait between scrape requests for non-OpenReview papers (default 3).",
    )
    args = parser.parse_args()

    structured_path = Path(f"data/final_output/{args.conference}/{args.year}/indian_papers_structured.json")
    if not structured_path.exists():
        parser.error(
            f"{structured_path} not found — run extraction (run_pipeline) for "
            f"{args.conference} {args.year} first."
        )

    papers = json.loads(structured_path.read_text(encoding="utf-8"))
    if args.max_papers:
        papers = papers[: args.max_papers]

    print(f"Fetching abstracts for {len(papers)} paper(s)...")

    def _progress(done, total):
        print(f"  [{done}/{total}] abstracts fetched", end="\r")

    papers_with_abstracts = fetch_abstracts_for_papers(
        papers, args.conference, args.year,
        refresh_cache=args.refresh_cache,
        delay_seconds=args.delay,
        on_progress=_progress,
    )
    print()

    failed = [p for p in papers_with_abstracts if not p.get("abstract")]
    if failed:
        print(f"  {len(failed)} paper(s) had no retrievable abstract:")
        for p in failed:
            print(f"    - {p.get('paper_title', 'Untitled')}: {p.get('error', 'unknown error')}")

    print("Summarizing...")
    summaries_by_index = SummaryLLM().summarize_all(papers_with_abstracts, args.conference, args.year)
    result = build_email(args.conference, args.year, papers_with_abstracts, summaries_by_index)

    print("\n" + "=" * 70)
    print(f"Subject: {result['subject']}")
    print("=" * 70)
    print(result["body"])
    print("=" * 70)
    if not result["intro_generated_by_llm"]:
        print(f"\n⚠️  Lead paragraph fell back to the templated one-liner. Reason: {result['intro_fallback_reason']}")
    print(f"\n{result['paper_count']} paper(s) included, {len(result['papers_skipped'])} skipped.")


if __name__ == "__main__":
    main()
