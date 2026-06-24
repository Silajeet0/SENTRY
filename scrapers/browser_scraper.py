"""
Tier 2: Playwright browser scraper.
Handles JS-rendered pages that Tier 1 cannot.

Works for: IEEE Xplore, ACM DL

NOTE ON HEADLESS MODE:
    ACM DL is protected by Cloudflare Turnstile which detects and blocks
    headless Chromium. The scraper runs headless=False for ACM URLs so
    Cloudflare passes the browser through. IEEE does not have this issue.

    ACM session cookies are loaded from data/acm_session_cookies.json
    (written by acm_api_fetcher.py during link extraction) so the
    Cloudflare clearance from the proceedings page is reused here,
    making per-paper scraping reliable without re-challenging.
"""
import re
import json
import time
import logging
from pathlib import Path
from urllib.parse import urlparse
from .base import BaseScraper, ScrapeResult
from utils.ieee_author_parser import (
    extract_author_section,
    extract_document_id,
    format_author_affiliation_pairs,
)

log = logging.getLogger(__name__)

ACM_COOKIE_PATH = Path("data/acm_session_cookies.json")
ACM_COOKIE_MAX_AGE_MINUTES = 30

# Domains that need JS rendering
BROWSER_DOMAINS = [
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "link.springer.com",
    "nature.com",
]

# Per-domain interaction rules before reading the DOM
DOMAIN_INTERACTIONS = {
    "ieeexplore.ieee.org": [
        ("click", "text=Authors"),
        ("wait", "1000"),
    ],
    "dl.acm.org": [
        ("scroll", "500"),
        ("wait", "500"),
    ],
}

PAYWALL_SIGNALS = [
    "access to this document requires a subscription",
    "purchase this article",
    "sign in to read",
    "buy this article",
    "institutional access required",
    "full text available to subscribers",
]


class BrowserScraper(BaseScraper):

    def can_handle(self, url: str) -> bool:
        return any(domain in url for domain in BROWSER_DOMAINS)

    def scrape(self, url: str) -> ScrapeResult:
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

            is_acm = "dl.acm.org" in url

            # ACM requires headless=False to pass Cloudflare Turnstile.
            # IEEE and others work fine headless.
            use_headless = not is_acm

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=use_headless,
                    args=["--disable-blink-features=AutomationControlled"]
                )

                context = browser.new_context(
                    viewport={"width": 1920, "height": 1080}
                )

                # Load ACM session cookies if fresh — avoids re-challenging
                # Cloudflare on every paper after link extraction
                if is_acm and ACM_COOKIE_PATH.exists():
                    age_minutes = (time.time() - ACM_COOKIE_PATH.stat().st_mtime) / 60
                    if age_minutes < ACM_COOKIE_MAX_AGE_MINUTES:
                        cookies = json.loads(ACM_COOKIE_PATH.read_text())
                        context.add_cookies(cookies)
                        log.debug(f"Loaded {len(cookies)} ACM session cookies ({age_minutes:.1f}m old)")
                    else:
                        log.warning(
                            f"ACM session cookies are {age_minutes:.0f}m old (>{ACM_COOKIE_MAX_AGE_MINUTES}m) "
                            "— Cloudflare may re-challenge. Re-run link extraction to refresh."
                        )

                page = context.new_page()

                page.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                """)

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(5000)

                    title = page.title()
                    log.debug(f"Page title: {title}")

                    # Detect Cloudflare challenge — fail fast rather than
                    # scraping the challenge page as if it were content
                    if "just a moment" in title.lower():
                        browser.close()
                        return ScrapeResult(
                            content="", source="browser", url=url,
                            success=False,
                            error="Cloudflare challenge — session cookies stale or missing"
                        )

                except PWTimeout:
                    browser.close()
                    return ScrapeResult(
                        content="", source="browser", url=url,
                        success=False, error="Page load timeout"
                    )

                # Domain-specific interactions
                domain = self._get_domain(url)
                for action, value in DOMAIN_INTERACTIONS.get(domain, []):
                    try:
                        if action == "click":
                            page.click(value, timeout=5000)
                        elif action == "wait":
                            page.wait_for_timeout(int(value))
                        elif action == "scroll":
                            page.evaluate(f"window.scrollBy(0, {value})")
                    except Exception:
                        pass

                # Extract citation meta tags
                metadata = page.evaluate("""
                    () => Array.from(document.querySelectorAll('meta[name^="citation_"]'))
                        .map(meta => {
                            const name = meta.getAttribute('name') || '';
                            const content = meta.getAttribute('content') || '';
                            return content ? `${name}: ${content}` : '';
                        })
                        .filter(Boolean)
                        .join('\\n')
                """)

                body_text = page.inner_text("body")
                content = "\n".join(part for part in [metadata, body_text] if part)

                # IEEE — navigate to /authors sub-page for affiliations
                if "ieeexplore.ieee.org/document/" in url:
                    authors_text = self._scrape_ieee_authors_page(page, url)
                    if authors_text:
                        content = (
                            "[IEEE authors and affiliations]\n"
                            + authors_text
                            + "\n\n[IEEE page metadata]\n"
                            + content
                        )

                # ACM — click Authors Info & Claims panel for affiliations,
                # extract only title + abstract (discard nav/footer noise)
                if "dl.acm.org/doi/10.1145/" in url:
                    authors_text = self._scrape_acm_authors_page(page, url)
                    acm_content = self._extract_acm_essentials(page)
                    if authors_text:
                        content = (
                            "[ACM authors and affiliations]\n"
                            + authors_text
                            + "\n\n[ACM paper content]\n"
                            + acm_content
                        )
                    else:
                        content = "[ACM paper content]\n" + acm_content

                browser.close()

            self._debug_dump(url, content)

            # Paywall check
            content_lower = content.lower()
            for signal in PAYWALL_SIGNALS:
                if signal in content_lower:
                    return ScrapeResult(
                        content="", source="browser", url=url,
                        success=False,
                        error=f"Paywall detected: '{signal}'"
                    )

            if len(content) < 200:
                return ScrapeResult(
                    content="", source="browser", url=url,
                    success=False, error="Page content too short"
                )

            return ScrapeResult(
                content=content[:15000],
                source="browser",
                url=url,
                success=True
            )

        except ImportError:
            return ScrapeResult(
                content="", source="browser", url=url,
                success=False,
                error="playwright not installed. Run: pip install playwright && playwright install chromium"
            )
        except Exception as e:
            return ScrapeResult(
                content="", source="browser", url=url,
                success=False, error=str(e)
            )

    # -----------------------------------------------------------------------
    # ACM helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_acm_essentials(page) -> str:
        """
        Extract only title, abstract, and index terms from ACM paper page.
        Discards navigation, cookie banners, references, and footer noise.
        """
        parts = []

        try:
            title = page.locator("h1.citation__title, h1").first.inner_text(timeout=3000)
            if title:
                parts.append(f"Title: {title.strip()}")
        except Exception:
            pass

        try:
            abstract = page.locator(
                "div.abstractSection p, "
                "section#abstract p, "
                "div[class*='abstract'] p"
            ).first.inner_text(timeout=3000)
            if abstract:
                parts.append(f"Abstract: {abstract.strip()}")
        except Exception:
            pass

        try:
            terms = page.locator(
                "ol.rlist--inline.comma li, "
                "div.article-terms span"
            ).all_inner_texts()
            if terms:
                parts.append(
                    f"Index Terms: {', '.join(t.strip() for t in terms if t.strip())}"
                )
        except Exception:
            pass

        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _parse_acm_contributors(raw: str) -> str:
        """
        Convert raw ACM contributor panel text into structured lines.

        Input:
            Contributor Metrics
            Expand All
            Ziwen Wang
            School of Computer Science, Anhui University, Hefei, China
            ...

        Output:
            Author: Ziwen Wang | Affiliation: School of Computer Science, Anhui University...
        """
        lines = [l.strip() for l in raw.splitlines() if l.strip()]

        skip = {"Contributor Metrics", "Expand All", "Collapse All"}
        lines = [l for l in lines if l not in skip]

        results = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Name heuristic: short, no commas, no digits, starts uppercase
            is_name = (
                len(line) < 60
                and "," not in line
                and not any(c.isdigit() for c in line)
                and bool(line) and line[0].isupper()
            )
            if is_name and i + 1 < len(lines):
                results.append(f"Author: {line} | Affiliation: {lines[i + 1]}")
                i += 2
            else:
                i += 1

        return "\n".join(results)

    @staticmethod
    def _scrape_acm_authors_page(page, url: str) -> str:
        """
        Click 'Authors Info & Claims' on the already-loaded paper page and
        extract author/affiliation pairs directly from the DOM.

        Strategy:
          1. Find the <a href="#tab-contributors"> link using its known stable
             attributes (data-id or class) and click it via JavaScript to avoid
             Playwright's default behaviour of waiting for navigation — the hash
             change is not a navigation but it can confuse the click() waiter.
          2. Wait for the contributors tab panel to become visible in the DOM
             using a targeted CSS selector rather than a broad body-text scan,
             which removes the race condition that caused garbled output.
          3. Extract name and affiliation directly from the panel's HTML nodes
             instead of parsing raw inner_text — this eliminates the name/
             affiliation swap entirely because we read structured data.
        """
        try:
            # Step 1: locate the "Authors Info & Claims" anchor.
            # We match on its known stable attributes in priority order.
            # Use JavaScript click so the hash change doesn't trigger
            # Playwright's navigation guard (which can cause a stale-page read).
            clicked = page.evaluate("""
                () => {
                    const selectors = [
                        'a[data-id="article-authors-viewall"]',
                        'a.to-authors-affiliations[href="#tab-contributors"]',
                        'a.to-authors-affiliations',
                    ];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (el) {
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }
            """)

            if not clicked:
                log.warning("ACM: 'Authors Info & Claims' link not found on page")
                return ""

            # Step 2: wait for the contributors tab panel to be present and
            # populated in the DOM. The panel has id="tab-contributors" and
            # contains li elements with author entries once rendered.
            # We do NOT rely on body.innerText to avoid reading a stale snapshot.
            try:
                page.wait_for_selector(
                    "#tab-contributors li.author-name, "
                    "#tab-contributors .author-info, "
                    "#tab-contributors .loa__author-name",
                    state="visible",
                    timeout=15000,
                )
            except Exception:
                # Fallback: wait for the panel itself to appear even if
                # the specific author-name selectors don't match the version
                # of ACM DL currently served.
                page.wait_for_selector(
                    "#tab-contributors",
                    state="visible",
                    timeout=10000,
                )
                page.wait_for_timeout(2000)  # let JS finish populating the panel

            # Step 3: extract structured name + affiliation pairs directly
            # from the DOM so there is no ambiguity about which text belongs
            # to a name vs an affiliation.
            pairs = page.evaluate("""
                () => {
                    const panel = document.querySelector('#tab-contributors');
                    if (!panel) return [];

                    const results = [];

                    // ACM DL renders each author as an <li> that contains:
                    //   .author-name (or .loa__author-name)  → the person's name
                    //   .author-info / .affiliation / p       → institution text
                    // We try several known selector patterns so the extractor
                    // is resilient to minor ACM DL markup version changes.

                    // Pattern A: .loa-authors list (most common as of 2024-25)
                    const loaItems = panel.querySelectorAll(
                        '.loa-authors li, .contributors__list li'
                    );
                    if (loaItems.length > 0) {
                        loaItems.forEach(li => {
                            const nameEl = li.querySelector(
                                '.author-name span, .loa__author-name, '
                                + '[class*="author-name"], strong'
                            );
                            const affEl = li.querySelector(
                                '.author-info, .affiliation, '
                                + '[class*="affiliation"], p'
                            );
                            const name = nameEl ? nameEl.innerText.trim() : '';
                            const aff  = affEl  ? affEl.innerText.trim()  : '';
                            if (name) results.push({ name, aff });
                        });
                        if (results.length > 0) return results;
                    }

                    // Pattern B: definition-list style (dl / dt / dd)
                    const dts = panel.querySelectorAll('dt');
                    if (dts.length > 0) {
                        dts.forEach(dt => {
                            const dd = dt.nextElementSibling;
                            const name = dt.innerText.trim();
                            const aff  = dd ? dd.innerText.trim() : '';
                            if (name) results.push({ name, aff });
                        });
                        if (results.length > 0) return results;
                    }

                    // Pattern C: generic fallback — grab every heading-like
                    // element followed by a paragraph inside the panel.
                    // This mirrors the old text heuristic but operates on
                    // structured nodes, not raw text, so order is guaranteed.
                    const children = Array.from(panel.children);
                    for (let i = 0; i < children.length - 1; i++) {
                        const tag = children[i].tagName;
                        if (['H2','H3','H4','STRONG','B'].includes(tag) ||
                            (children[i].className || '').includes('author')) {
                            const name = children[i].innerText.trim();
                            const aff  = children[i + 1].innerText.trim();
                            if (name && aff) results.push({ name, aff });
                        }
                    }
                    return results;
                }
            """)

            if not pairs:
                log.warning("ACM: contributors panel found but no author pairs extracted")
                return ""

            lines = [
                f"Author: {p['name']} | Affiliation: {p['aff']}"
                for p in pairs
                if p.get("name")
            ]
            return "\n".join(lines)[:12000]

        except Exception as e:
            log.warning(f"ACM authors panel scrape failed: {e}")
            return ""

    # -----------------------------------------------------------------------
    # IEEE helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _scrape_ieee_authors_page(page, url: str) -> str:
        """
        Navigate to /authors#authors sub-page which exposes author-affiliation
        pairs without requiring full text access.
        """
        document_id = extract_document_id(url)
        if not document_id:
            return ""

        authors_url = (
            f"https://ieeexplore.ieee.org/document/{document_id}/authors#authors"
        )
        try:
            page.goto(authors_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2500)

            for selector in [
                "a:has-text('All Authors')",
                "button:has-text('Authors')",
                "text=Authors",
            ]:
                try:
                    locator = page.locator(selector).first
                    locator.scroll_into_view_if_needed(timeout=3000)
                    locator.click(timeout=3000)
                    page.wait_for_timeout(1000)
                    break
                except Exception:
                    pass

            text = page.inner_text("body", timeout=10000)
            section = extract_author_section(text)
            pairs = format_author_affiliation_pairs(section)
            return pairs or section[:12000]
        except Exception:
            return ""

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_domain(url: str) -> str:
        match = re.search(r"https?://([^/]+)", url)
        return match.group(1) if match else ""

    @staticmethod
    def _debug_dump(url: str, content: str) -> None:
        debug_dir = Path("data/debug_scrapes")
        if not debug_dir.exists():
            return
        parsed = urlparse(url)
        document_match = re.search(r"/document/(\d+)", parsed.path)
        name = (
            document_match.group(1)
            if document_match
            else re.sub(r"\W+", "_", url)[:80]
        )
        path = debug_dir / f"{name}.txt"
        path.write_text(content, encoding="utf-8")
