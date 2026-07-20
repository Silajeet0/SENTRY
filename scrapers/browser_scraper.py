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

NOTE ON OPENREVIEW — deliberately NOT handled here:
    OpenReview-hosted conferences go through
    workflows/link_extractors/openreview_api_fetcher.py's authenticated
    API path (pipeline.process_openreview_paper) instead — title, abstract,
    and ground-truth author/institution data come straight from the API,
    with no scraping, no browser, no Cloudflare-style challenge handling
    needed. openreview.net is intentionally absent from BROWSER_DOMAINS
    below so an openreview.net URL can never silently land here (a caller
    bug routing one to process_paper() instead of process_openreview_paper()
    will cleanly fail "all tiers failed" instead of quietly running the old,
    now-unsupported cookie-reuse-and-PDF-download approach).
"""
import re
import json
import logging
from pathlib import Path
from urllib.parse import urlparse
from .base import BaseScraper, ScrapeResult
from utils.ieee_author_parser import (
    extract_author_section,
    extract_document_id,
    format_author_affiliation_pairs,
)
from workflows.link_extractors.acm_api_fetcher import warmup_acm_cookies

log = logging.getLogger(__name__)

ACM_COOKIE_PATH = Path("data/acm_session_cookies.json")

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
    
    @staticmethod
    def _is_challenge_title(title: str) -> bool:
        t = title.lower()
        return "just a moment" in t or "challenge" in t

    @staticmethod
    def _acm_proceedings_url(paper_url: str) -> str | None:
        m = re.search(r"/doi/10\.1145/(\d+)\.\d+", paper_url)
        if not m:
            return None
        return f"https://dl.acm.org/doi/proceedings/10.1145/{m.group(1)}"

    def scrape(self, url: str, _retry: bool = False) -> ScrapeResult:
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

            is_acm = "dl.acm.org" in url

            # ACM requires headless=False — its bot challenge wall detects
            # and blocks headless Chromium. IEEE and others work fine headless.
            use_headless = False

            # NOTE: cookie freshness is checked reactively, not on a wall-clock
            # timer. We try the scrape with whatever cookies are on disk (even
            # if they're old — they may well still be valid), and only pay the
            # cost of a full Cloudflare re-clearance if we actually get
            # challenged (see the title check below). This avoids the old
            # behaviour of unconditionally re-warming every ~30 minutes even
            # when the existing session was still perfectly good.

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
                    cookies = json.loads(ACM_COOKIE_PATH.read_text())
                    context.add_cookies(cookies)
                    log.debug(f"Loaded {len(cookies)} ACM session cookies")

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

                    # Detect bot challenge. Covers Cloudflare's
                    # "Just a moment..." (ACM). For ACM specifically, this is
                    # the ONE place a cookie refresh gets triggered — reactively,
                    # because we just proved the stored session no longer
                    # clears Cloudflare — rather than pre-emptively on a timer.
                    if self._is_challenge_title(title):
                        browser.close()

                        if is_acm and not _retry:
                            log.warning(
                                f"ACM bot challenge detected (title: {title!r}) — "
                                "stored session cookies no longer clear Cloudflare. "
                                "Refreshing session and retrying once."
                            )
                            base_url = self._acm_proceedings_url(url)
                            if not base_url:
                                return ScrapeResult(
                                    content="", source="browser", url=url,
                                    success=False,
                                    error=(
                                        "Bot challenge detected and could not derive "
                                        "the ACM proceedings URL to refresh cookies from"
                                    ),
                                )
                            try:
                                warmup_acm_cookies(base_url)
                            except Exception as e:
                                log.warning(f"ACM cookie refresh failed: {e}")
                                return ScrapeResult(
                                    content="", source="browser", url=url,
                                    success=False,
                                    error=f"Bot challenge detected; cookie refresh failed: {e}",
                                )
                            return self.scrape(url, _retry=True)

                        return ScrapeResult(
                            content="", source="browser", url=url,
                            success=False,
                            error=(
                                "Bot challenge detected even after a session refresh"
                                if _retry else
                                "Bot challenge detected — session cookies stale or missing"
                            ),
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

                # OpenReview intentionally not handled here — see module
                # docstring. openreview.net is absent from BROWSER_DOMAINS,
                # so this branch is unreachable in normal operation.

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
        Extract author/affiliation pairs directly from the live DOM using
        page.evaluate(), with a polling wait until affiliations are populated.

        Key insight: the affiliation divs are injected by JS after page load.
        page.content() misses them. outerHTML via evaluate() sees them — but
        only after the JS has run. We poll until at least one affiliation span
        has non-empty text, then extract everything at once.
        """
        try:
            # Poll until affiliation spans are populated (max 15s).
            # The JS that fills them runs asynchronously after page load.
            for _ in range(15):
                has_affs = page.evaluate("""
                    () => {
                        const spans = document.querySelectorAll(
                            'div[property="affiliation"] span[property="name"]'
                        );
                        return [...spans].some(s => s.innerText.trim().length > 0);
                    }
                """)
                if has_affs:
                    break
                page.wait_for_timeout(1000)
            else:
                log.warning("ACM: affiliation spans never populated after 15s")

            # Extract directly from the live DOM.
            # DOM structure confirmed from DevTools:
            #   span[property="author"]    — one per author, names only, flat list
            #   div.affiliations           — one per author, in matching document order
            #     div[property="affiliation"]  — ONE OR MORE per author
            #       span[property="name"]      — affiliation text
            #
            # Names and affiliation groups are zipped by index.
            # Multiple affiliations for one author are joined with "; ".
            pairs = page.evaluate("""
                () => {
                    // Author names in document order
                    const names = [...document.querySelectorAll(
                        'span[property="author"][typeof="Person"]'
                    )].map(span => {
                        const given  = span.querySelector('span[property="givenName"]');
                        const family = span.querySelector('span[property="familyName"]');
                        if (given || family) {
                            return [given, family]
                                .filter(Boolean)
                                .map(el => (el.textContent || '').trim())
                                .join(' ').trim();
                        }
                        const a = span.querySelector('a[title]');
                        return a ? a.getAttribute('title').trim() : '';
                    }).filter(Boolean);

                    // Affiliation groups in document order.
                    // Each div.affiliations = one author's affiliation block.
                    // Multiple div[property="affiliation"] inside = multiple affs.
                    const affGroups = [...document.querySelectorAll('div.affiliations')]
                        .map(group => {
                            const texts = [...group.querySelectorAll(
                                'div[property="affiliation"][typeof="Organization"] '
                                + 'span[property="name"]'
                            )].map(s => (s.textContent || '').trim()).filter(Boolean);
                            return texts.join('; ') || 'Unknown';
                        });

                    // Zip by index
                    return names.map((name, i) => ({
                        name,
                        aff: affGroups[i] || 'Unknown'
                    }));
                }
            """)

            if not pairs:
                log.warning("ACM: no author/affiliation pairs found in live DOM")
                return ""

            lines = [
                f"Author: {p['name']} | Affiliation: {p['aff']}"
                for p in pairs if p.get("name")
            ]
            log.debug(f"ACM live DOM: extracted {len(lines)} authors")
            return "\n".join(lines)[:12000]

        except Exception as e:
            log.warning(f"ACM live DOM extraction failed: {e}")
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