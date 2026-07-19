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

NOTE ON PERSISTENT BROWSER (why scrape() no longer launches Chromium fresh
every call):
    A brand-new headless=False Chromium process + a brand-new TLS handshake
    to dl.acm.org, repeated every ~10-15s for an hour straight, is a very
    regular, very fingerprintable traffic pattern — independent of raw
    request volume, this is itself a strong bot signal. One Chromium
    process + one browser context is now launched lazily on first use and
    reused for every subsequent scrape() call (only a fresh Page is opened
    and closed per paper, which is cheap and keeps one paper's DOM state
    from leaking into the next). This also means cookies accumulated during
    the session persist naturally in the context's own cookie jar, on top
    of the existing load-from-disk logic below — closer to how a real
    browsing session actually behaves.

    The browser is intentionally NOT torn down between conferences/runs —
    under the orchestrator (a long-lived process), this lets one browser
    process serve the entire session's ACM scraping rather than relaunching
    per run. call .close() explicitly if you want to release it (e.g. at
    full process shutdown); if the browser/context ever dies unexpectedly
    (crash, killed process), _ensure_browser() detects that and relaunches
    once automatically rather than failing every subsequent call.

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
from workflows.link_extractors.acm_api_fetcher import warmup_acm_cookies

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

    def __init__(self):
        # Lazily launched on first scrape() call, then reused — see module
        # docstring "NOTE ON PERSISTENT BROWSER" above.
        self._playwright = None
        self._browser = None
        self._context = None

    def can_handle(self, url: str) -> bool:
        return any(domain in url for domain in BROWSER_DOMAINS)
    
    @staticmethod
    def _acm_proceedings_url(paper_url: str) -> str | None:
        m = re.search(r"/doi/10\.1145/(\d+)\.\d+", paper_url)
        if not m:
            return None
        return f"https://dl.acm.org/doi/proceedings/10.1145/{m.group(1)}"

    def _load_acm_cookies_into_context(self) -> None:
        """Best-effort refresh of ACM cookies into the persistent context.
        Cheap even when nothing changed — safe to call before every ACM
        scrape (matches the original per-call behavior), and picks up a
        newly-warmed cookie file mid-run without needing a browser relaunch."""
        if ACM_COOKIE_PATH.exists():
            try:
                cookies = json.loads(ACM_COOKIE_PATH.read_text())
                self._context.add_cookies(cookies)
                log.debug(f"Loaded {len(cookies)} ACM session cookies into persistent context")
            except Exception as e:
                log.warning(f"Failed to load ACM cookies into context: {e}")

    def _ensure_browser(self):
        """
        Lazily launch the persistent Chromium instance on first use, or
        after it's been closed/died. Reused across every subsequent
        scrape() call — see module docstring "NOTE ON PERSISTENT BROWSER".
        """
        if self._browser is not None:
            try:
                # is_connected() is the cheap way to detect a browser that
                # crashed or was killed out from under us since last use.
                if self._browser.is_connected():
                    return
                log.warning("Persistent browser disconnected — relaunching.")
            except Exception:
                log.warning("Persistent browser in a bad state — relaunching.")
            self.close()

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        # headless=False unconditionally (unchanged from before) — ACM's
        # bot challenge wall detects and blocks headless Chromium; running
        # every domain this way is simplest and already what the code did.
        self._browser = self._playwright.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        self._context = self._browser.new_context(
            viewport={"width": 1920, "height": 1080}
        )
        self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        self._load_acm_cookies_into_context()
        log.info("Launched persistent browser instance (reused across papers).")

    def close(self) -> None:
        """Explicitly release the persistent browser. Not called between
        papers or between runs by design — only when a caller genuinely
        wants to release resources (e.g. full process shutdown), or
        internally by _ensure_browser() when recovering from a dead browser."""
        for closer, obj in (
            ("context", self._context),
            ("browser", self._browser),
        ):
            try:
                if obj:
                    obj.close()
            except Exception as e:
                log.debug(f"Error closing {closer} (non-fatal): {e}")
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            log.debug(f"Error stopping playwright (non-fatal): {e}")
        self._context = None
        self._browser = None
        self._playwright = None

    def scrape(self, url: str) -> ScrapeResult:
        page = None
        try:
            from playwright.sync_api import TimeoutError as PWTimeout

            is_acm = "dl.acm.org" in url

            if is_acm and ACM_COOKIE_PATH.exists():
                age_minutes = (time.time() - ACM_COOKIE_PATH.stat().st_mtime) / 60
                if age_minutes >= ACM_COOKIE_MAX_AGE_MINUTES:
                    log.warning(
                        f"ACM session cookies are {age_minutes:.0f}m old "
                        f"(>{ACM_COOKIE_MAX_AGE_MINUTES}m) — refreshing before scrape."
                    )
                    base_url = self._acm_proceedings_url(url)
                    if base_url:
                        try:
                            warmup_acm_cookies(base_url)
                        except Exception as e:
                            log.warning(f"ACM cookie warmup failed: {e}")
                    else:
                        log.warning(f"Could not derive ACM proceedings URL from {url}")

            self._ensure_browser()

            # Refresh cookies into the persistent context before every ACM
            # scrape (cheap; picks up any warmup_acm_cookies() write above,
            # or one triggered by an earlier paper this run, without a
            # browser relaunch).
            if is_acm:
                self._load_acm_cookies_into_context()

            page = self._context.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(5000)

                title = page.title()
                log.debug(f"Page title: {title}")

                # Detect bot challenge — fail fast rather than scraping
                # the challenge page as if it were content. Covers
                # Cloudflare's "Just a moment..." (ACM).
                if "just a moment" in title.lower() or "challenge" in title.lower():
                    return ScrapeResult(
                        content="", source="browser", url=url,
                        success=False,
                        error="Bot challenge detected — session cookies stale or missing"
                    )

            except PWTimeout:
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
            # If the persistent browser died mid-scrape (crashed, killed,
            # connection dropped), close() so the NEXT call's
            # _ensure_browser() relaunches fresh instead of every
            # subsequent paper failing against a dead handle.
            err_text = str(e).lower()
            if any(s in err_text for s in ("target closed", "browser has been closed", "connection closed")):
                log.warning(f"Persistent browser appears dead ({e}) — will relaunch on next scrape.")
                self.close()
            return ScrapeResult(
                content="", source="browser", url=url,
                success=False, error=str(e)
            )
        finally:
            # Only the page is per-paper — the browser/context persist.
            if page is not None:
                try:
                    page.close()
                except Exception:
                    pass

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