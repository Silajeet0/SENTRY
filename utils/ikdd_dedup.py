"""
ikdd_dedup.py — IKDD backend deduplication checker.

Scrapes all approved Premier Papers from ikdd backend and checks
whether a candidate paper (by title) is already present.

"""

import os
import json
import logging
import argparse
import unicodedata
import re
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

# Paths

IKDD_BASE          = "https://ikdd.acm.org"
LOGIN_URL          = f"{IKDD_BASE}/ikdd-login.php"
APPROVED_LIST_URL  = f"{IKDD_BASE}/premier-papers-list.php?status=2"

CACHE_PATH         = Path("data/ikdd_cache/approved_titles.json")

# Similarity threshold: 0.0–1.0. Titles scoring above this are considered duplicates. Jaccard Similarity is used
SIMILARITY_THRESHOLD = 0.85

# Title normalisation
def _normalise(title: str) -> str:
    """
    Lowercase, strip accents, collapse whitespace, remove punctuation.
    """
    # Unicode normalise (decompose accented chars)
    title = unicodedata.normalize("NFD", title)
    title = "".join(c for c in title if unicodedata.category(c) != "Mn")
    title = title.lower()
    # Remove punctuation except alphanumerics and spaces
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _similarity(a: str, b: str) -> float:
    """
    Token-based Jaccard similarity between two normalised titles.
    """
    tokens_a = set(_normalise(a).split())
    tokens_b = set(_normalise(b).split())
    if not tokens_a or not tokens_b:
        return 1.0 if a == b else 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# scraper
def _scrape_approved_titles(username: str, password: str) -> list[str]:
    """
    Logs in to IKDD and scrapes all approved Premier Paper titles.
    Returns a list of raw title strings as they appear in the UI.
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        raise RuntimeError(
            "playwright is required. "
            "Install with: pip install playwright && playwright install chromium"
        )

    titles = []

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

        # Login
        log.info(f"Navigating to login: {LOGIN_URL}")
        page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)

        # Fill credentials
        # attributes adjusted according to the actual DOM of the login page
        page.fill("input[name='username'], input[type='email'], input[name='email']", username)
        page.fill("input[name='password'], input[type='password']", password)
        page.click("input[type='button'], button[type='login']")

        try:
            # Wait for post-login redirect — dashboard has "indexDashboard" in URL
            page.wait_for_url("**/indexDashboard**", timeout=15000)
            log.info("Login successful")
        except PWTimeout:
            # if redirected to a different URL — check we're no longer on login, this is a failsafe
            current = page.url
            if "ikdd-login.php" in current or "login" in current.lower():
                browser.close()
                raise RuntimeError(
                    "Login failed — still on login page. "
                    "Check IKDD_USERNAME and IKDD_PASSWORD in your .env"
                )
            log.info(f"Login succeeded (redirected to {current})")

        # Navigate to approved list of premier papers
        log.info(f"Fetching approved papers: {APPROVED_LIST_URL}")
        page.goto(APPROVED_LIST_URL, wait_until="domcontentloaded", timeout=30000)

        # select 100 papers per page; if unavailable fall back to pagination (failsafe for future).
        try:
            select = page.locator("select").first
            select.select_option(value="100")
            page.wait_for_timeout(1000)
            log.info("Set rows per page to 100")
        except Exception:
            log.warning("Could not set rows-per-page dropdown — will paginate")

        # Paginate and collect title
        page_num = 1
        last_page_first_title = None  # content-change sentinel

        while True:
            log.info(f"Scraping page {page_num}...")

            # Wait for table rows to be present
            page.wait_for_selector("table tbody tr", timeout=15000)

            rows = page.locator("table tbody tr")
            count = rows.count()
            log.info(f"  Found {count} rows on page {page_num}")

            page_titles = []
            for i in range(count):
                row = rows.nth(i)
                cells = row.locator("td")
                if cells.count() < 2:
                    continue
                title_cell = None
                for j in range(cells.count()):
                    cell = cells.nth(j)
                    text = (cell.inner_text() or "").strip()
                    if len(text) > 10 and not text.isdigit() and not cell.locator("input").count():
                        title_cell = text
                        break
                if title_cell:
                    page_titles.append(title_cell)

            # Content-change sentinel: if the first title on this page matches
            # the first title from the previous page, DataTables didn't advance
            # we've hit the end and should stop
            current_first = page_titles[0] if page_titles else None
            if page_num > 1 and current_first == last_page_first_title:
                log.info(f"Page content unchanged — reached last page after {page_num - 1} real pages.")
                break

            last_page_first_title = current_first
            titles.extend(page_titles)

            # DataTables wraps the Next button in a <li class="next disabled"> (identified from DOM)
            # when on the last page. is_enabled() is unreliable — check the
            # parent <li> for the "disabled" class instead.
            next_li = page.locator("li.next")
            if next_li.count() > 0:
                li_class = next_li.first.get_attribute("class") or ""
                if "disabled" in li_class:
                    log.info(f"Next button disabled — no more pages. Total: {len(titles)}")
                    break
                # Click the <a> inside the <li>
                next_li.first.locator("a").click()
                page.wait_for_timeout(1500)
                page_num += 1
            else:
                # Fallback: try generic Next button text
                next_btn = page.locator("a:has-text('Next'), button:has-text('Next')")
                if next_btn.count() > 0:
                    next_btn.first.click()
                    page.wait_for_timeout(1500)
                    page_num += 1
                else:
                    log.info(f"No Next button found — done. Total: {len(titles)}")
                    break

        browser.close()

    return titles



# Cache management
def refresh_cache(username: str = None, password: str = None) -> dict:
    """
    Scrapes IKDD and saves approved titles to local cache.
    Returns the cache dict.
    """
    username = username or os.getenv("IKDD_USERNAME", "")
    password = password or os.getenv("IKDD_PASSWORD", "")

    if not username or not password:
        raise ValueError(
            "IKDD credentials not found. "
            "Set IKDD_USERNAME and IKDD_PASSWORD in your .env file."
        )

    log.info("Refreshing IKDD approved titles cache...")
    titles = _scrape_approved_titles(username, password)

    cache = {
        "fetched_at": datetime.now().isoformat(),
        "total": len(titles),
        "titles": titles,
        # Pre-compute normalised versions for fast lookup
        "normalised": [_normalise(t) for t in titles],
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Cache saved: {len(titles)} approved titles → {CACHE_PATH}")
    return cache


def load_cache() -> dict:
    """Load the local cache. Raises if not found — call refresh_cache() first."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"IKDD cache not found at {CACHE_PATH}. "
            "Run with --refresh first: python -m utils.ikdd_dedup --refresh"
        )
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

# Deduplication API
class IKDDDeduplicator:
    """
    Main deduplication interface. Call is_duplicate() per paper.
    """

    def __init__(self, cache: dict = None):
        """
        Args:
            cache: Pre-loaded cache dict. If None, loads from disk.
        """
        self._cache = cache or load_cache()
        self._normalised = self._cache.get("normalised", [
            _normalise(t) for t in self._cache["titles"]
        ])
        self._titles = self._cache["titles"]
        log.info(
            f"IKDDDeduplicator ready — {len(self._titles)} approved titles "
            f"(cache from {self._cache.get('fetched_at', 'unknown')})"
        )

    def is_duplicate(self, candidate_title: str) -> "DedupResult":
        """
        Check if candidate_title is already in IKDD.

        Returns DedupResult with:
            .is_duplicate  bool
            .score         float — best similarity score found
            .matched_title str   — the IKDD title it matched against (or "")
        """
        norm_candidate = _normalise(candidate_title)

        best_score = 0.0
        best_title = ""

        for i, norm_existing in enumerate(self._normalised):
            score = _similarity(norm_candidate, norm_existing)
            if score > best_score:
                best_score = score
                best_title = self._titles[i]
            # Short-circuit on near-exact match
            if best_score >= 0.99:
                break

        is_dup = best_score >= SIMILARITY_THRESHOLD
        return DedupResult(
            candidate=candidate_title,
            is_duplicate=is_dup,
            score=best_score,
            matched_title=best_title if is_dup else "",
        )

    def filter_new(self, papers: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Split a list of paper dicts into (new_papers, duplicates).

        Expects each paper dict to have a "paper_title" key,
        matching the structure of indian_papers_structured.json.

        Returns:
            new_papers  — papers not in IKDD (safe to push)
            duplicates  — papers already in IKDD (skip)
        """
        new_papers = []
        duplicates = []

        for paper in papers:
            title = paper.get("paper_title", "")
            if not title:
                log.warning(f"Paper with no title — skipping dedup check: {paper.get('paper_url')}")
                new_papers.append(paper)
                continue

            result = self.is_duplicate(title)
            if result.is_duplicate:
                log.info(
                    f"DUPLICATE ({result.score:.2f}): '{title}' "
                    f"→ matched '{result.matched_title}'"
                )
                paper["_dedup_status"] = "duplicate"
                paper["_dedup_score"] = round(result.score, 4)
                paper["_dedup_matched"] = result.matched_title
                duplicates.append(paper)
            else:
                paper["_dedup_status"] = "new"
                new_papers.append(paper)

        return new_papers, duplicates


from dataclasses import dataclass

@dataclass
class DedupResult:
    candidate: str
    is_duplicate: bool
    score: float
    matched_title: str


# CLI

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    parser = argparse.ArgumentParser(
        description="IKDD deduplication checker for AEGIS pipeline outputs."
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Scrape IKDD and refresh the local approved titles cache."
    )
    parser.add_argument(
        "--check", type=str, metavar="TITLE",
        help="Check a single paper title against the cache."
    )
    parser.add_argument(
        "--check-file", type=str, metavar="PATH",
        help="Check all papers in an indian_papers_structured.json file."
    )
    parser.add_argument(
        "--threshold", type=float, default=SIMILARITY_THRESHOLD,
        help=f"Similarity threshold for duplicate detection (default: {SIMILARITY_THRESHOLD})"
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    if args.refresh:
        cache = refresh_cache()
        print(f"\n✅ Cache refreshed — {cache['total']} approved titles stored.")
        return

    if not args.check and not args.check_file:
        parser.print_help()
        return

    # Load cache for check operations
    try:
        dedup = IKDDDeduplicator()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return

    if args.check:
        result = dedup.is_duplicate(args.check)
        if result.is_duplicate:
            print(f"\n⚠️  DUPLICATE (score: {result.score:.2f})")
            print(f"   Your title : {result.candidate}")
            print(f"   IKDD title : {result.matched_title}")
        else:
            print(f"\n✅ NEW — not found in IKDD (best score: {result.score:.2f})")

    if args.check_file:
        path = Path(args.check_file)
        if not path.exists():
            print(f"\n❌ File not found: {path}")
            return

        papers = json.loads(path.read_text(encoding="utf-8"))
        print(f"\nChecking {len(papers)} papers against IKDD approved list...")

        new_papers, duplicates = dedup.filter_new(papers)

        print(f"\n{'─'*50}")
        print(f"  Total papers  : {len(papers)}")
        print(f"  New (safe)    : {len(new_papers)}")
        print(f"  Duplicates    : {len(duplicates)}")
        print(f"{'─'*50}")

        if duplicates:
            print(f"\n⚠️  Duplicates found:")
            for p in duplicates:
                print(f"  [{p['_dedup_score']:.2f}] {p['paper_title']}")
                print(f"         → {p['_dedup_matched']}")

        # Save filtered output alongside original
        out_new = path.parent / "indian_papers_new_only.json"
        out_dups = path.parent / "indian_papers_duplicates.json"
        out_new.write_text(json.dumps(new_papers, indent=2, ensure_ascii=False), encoding="utf-8")
        out_dups.write_text(json.dumps(duplicates, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n  New papers saved     → {out_new}")
        print(f"  Duplicates saved     → {out_dups}")


if __name__ == "__main__":
    main()
