"""
Generate deterministic ground-truth labels from IEEE Xplore authors pages.

This does not call an LLM. It reads /document/<id>/authors#authors, extracts
author-affiliation pairs, and applies factual affiliation rules.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from evaluation.india_rules import classify_affiliation, combine_author_decisions
from utils.ieee_author_parser import (
    extract_author_affiliation_pairs,
    extract_author_section,
    extract_document_id,
    normalize_ieee_url,
)


def label_paper(url: str, page_text: str) -> dict:
    section = extract_author_section(page_text)
    authors = []
    for author in extract_author_affiliation_pairs(section):
        decision = classify_affiliation(author["affiliation"])
        authors.append({
            **author,
            "label": decision.label,
            "matched_rules": decision.matches,
        })
    decisions = [classify_affiliation(author["affiliation"]) for author in authors]
    label = combine_author_decisions(decisions) if decisions else "unknown"

    indian_authors = [
        author["name"] for author in authors
        if author["label"] == "positive"
    ]
    indian_institutions = sorted({
        author["affiliation"] for author in authors
        if author["label"] == "positive"
    })

    return {
        "paper_url": normalize_ieee_url(url),
        "document_id": extract_document_id(url),
        "label": label,
        "is_indian_affiliated": label == "positive",
        "authors": authors,
        "indian_authors": indian_authors,
        "indian_institutions": indian_institutions,
        "raw_author_section": section,
    }


def scrape_ieee_authors_page(page, url: str) -> str:
    document_id = extract_document_id(url)
    if not document_id:
        raise ValueError(f"Could not extract IEEE document id from URL: {url}")

    authors_url = f"https://ieeexplore.ieee.org/document/{document_id}/authors#authors"
    page.goto(authors_url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(2500)

    for selector in ["a:has-text('All Authors')", "button:has-text('Authors')", "text=Authors"]:
        try:
            locator = page.locator(selector).first
            locator.scroll_into_view_if_needed(timeout=3000)
            locator.click(timeout=3000)
            page.wait_for_timeout(1000)
            break
        except Exception:
            pass

    return page.inner_text("body", timeout=10000)


def save_outputs(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ground_truth.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for label in ["positive", "negative", "ambiguous", "unknown"]:
        subset = [record for record in records if record["label"] == label]
        (output_dir / f"{label}.json").write_text(
            json.dumps(subset, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = {
        "total": len(records),
        "positive": sum(1 for r in records if r["label"] == "positive"),
        "negative": sum(1 for r in records if r["label"] == "negative"),
        "ambiguous": sum(1 for r in records if r["label"] == "ambiguous"),
        "unknown": sum(1 for r in records if r["label"] == "unknown"),
        "generated_at": datetime.now().isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic IEEE ground-truth labels.")
    parser.add_argument("--links", required=True, help="Path to links.json.")
    parser.add_argument("--conference", default="IEEE-ICDM")
    parser.add_argument("--year", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    links = json.loads(Path(args.links).read_text(encoding="utf-8"))
    if args.max_papers:
        links = links[:args.max_papers]

    output_dir = Path(args.output_dir or f"data/ground_truth/{args.conference}/{args.year}")
    records = []
    if args.resume and (output_dir / "ground_truth.json").exists():
        records = json.loads((output_dir / "ground_truth.json").read_text(encoding="utf-8"))

    processed = {record["paper_url"] for record in records}

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        for index, url in enumerate(links, start=1):
            normalized_url = normalize_ieee_url(url)
            if normalized_url in processed:
                print(f"[{index}/{len(links)}] skip {normalized_url}")
                continue

            print(f"[{index}/{len(links)}] label {normalized_url}")
            try:
                page_text = scrape_ieee_authors_page(page, normalized_url)
                record = label_paper(normalized_url, page_text)
            except Exception as e:
                record = {
                    "paper_url": normalized_url,
                    "document_id": extract_document_id(normalized_url),
                    "label": "unknown",
                    "is_indian_affiliated": False,
                    "authors": [],
                    "indian_authors": [],
                    "indian_institutions": [],
                    "raw_author_section": "",
                    "error": str(e),
                }

            records.append(record)
            save_outputs(records, output_dir)
            time.sleep(args.delay)

        browser.close()

    save_outputs(records, output_dir)
    print(f"Saved ground truth to: {output_dir / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
