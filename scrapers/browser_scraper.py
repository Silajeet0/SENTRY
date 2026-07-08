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
from workflows.link_extractors.acm_api_fetcher import warmup_acm_cookies

log = logging.getLogger(__name__)

ACM_COOKIE_PATH = Path("data/acm_session_cookies.json")
ACM_COOKIE_MAX_AGE_MINUTES = 30

OPENREVIEW_COOKIE_PATH = Path("data/openreview_session_cookies.json")
OPENREVIEW_COOKIE_MAX_AGE_MINUTES = 30

# Domains that need JS rendering
BROWSER_DOMAINS = [
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "link.springer.com",
    "nature.com",
    "openreview.net",
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
    def _acm_proceedings_url(paper_url: str) -> str | None:
        m = re.search(r"/doi/10\.1145/(\d+)\.\d+", paper_url)
        if not m:
            return None
        return f"https://dl.acm.org/doi/proceedings/10.1145/{m.group(1)}"

    def scrape(self, url: str) -> ScrapeResult:
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

            is_acm = "dl.acm.org" in url
            is_openreview = "openreview.net" in url

            # ACM and OpenReview both require headless=False — their bot
            # challenge walls detect and block headless Chromium. IEEE and
            # others work fine headless.
            use_headless = False
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

                # Load OpenReview session cookies if fresh — same Cloudflare-
                # style challenge wall as ACM, same cookie-reuse strategy.
                if is_openreview and OPENREVIEW_COOKIE_PATH.exists():
                    age_minutes = (time.time() - OPENREVIEW_COOKIE_PATH.stat().st_mtime) / 60
                    if age_minutes < OPENREVIEW_COOKIE_MAX_AGE_MINUTES:
                        or_cookies = json.loads(OPENREVIEW_COOKIE_PATH.read_text())
                        context.add_cookies(or_cookies)
                        log.debug(f"Loaded {len(or_cookies)} OpenReview session cookies ({age_minutes:.1f}m old)")
                    else:
                        log.warning(
                            f"OpenReview session cookies are {age_minutes:.0f}m old "
                            f"(>{OPENREVIEW_COOKIE_MAX_AGE_MINUTES}m) — challenge may "
                            "reappear. Re-run link extraction to refresh."
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

                    # Detect bot challenge — fail fast rather than scraping
                    # the challenge page as if it were content. Covers both
                    # Cloudflare's "Just a moment..." (ACM) and OpenReview's
                    # challenge page title.
                    if "just a moment" in title.lower() or "challenge" in title.lower():
                        browser.close()
                        return ScrapeResult(
                            content="", source="browser", url=url,
                            success=False,
                            error="Bot challenge detected — session cookies stale or missing"
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

                # OpenReview — download the PDF directly via requests using
                # saved session cookies,
                # then extract page 1 text for title/authors/affiliations.
                # This replaces `content` rather than appending — the forum
                # page's body text (reviews, discussion threads) is mostly
                # noise for author/affiliation extraction.
                if "openreview.net/forum" in url:
                    pdf_text = self._scrape_openreview_pdf(page, url)
                    if pdf_text:
                        content = (
                            "[OpenReview PDF page 1 — title, authors, affiliations]\n"
                            + pdf_text
                        )

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

    @staticmethod
    def _scrape_openreview_pdf(page, url: str) -> str:
        """
        Download the OpenReview PDF via requests using saved session cookies,
        and extract page 1 text (title, authors, affiliations) with pdfminer.

        Why requests + cookies instead of context.route()
        ─────────────────────────────────────────────────
        The route-intercept approach (route.fetch() inside a Playwright route
        handler) is fragile: it makes the request from Node.js's networking
        stack, not the browser's, so it doesn't reliably carry the page's
        cookie jar or TLS fingerprint. In practice this caused two failure
        modes — empty captures, and an asyncio race where the browser context
        closed mid-route-handler, crashing with TargetClosedError.

        The fix mirrors what worked for ACM DL: download with plain requests,
        passing the session cookies saved to disk (OPENREVIEW_COOKIE_PATH) by
        the Playwright session that already cleared any challenge wall. This
        is synchronous, has no route lifecycle to manage, and cannot race
        against the browser closing because it doesn't touch the browser at all.

        Forum URL → PDF URL:
            openreview.net/forum?id=X  →  openreview.net/pdf?id=X
        """
        try:
            import io
            import requests
            from pdfminer.high_level import extract_text

            pdf_url = url.replace("openreview.net/forum", "openreview.net/pdf", 1)
            log.debug(f"OpenReview PDF URL: {pdf_url}")

            # Load saved session cookies — captured during link extraction or
            # warmup, mirrors ACM's cookie reuse pattern exactly.
            session_cookies: dict[str, str] = {}
            if OPENREVIEW_COOKIE_PATH.exists():
                age_minutes = (time.time() - OPENREVIEW_COOKIE_PATH.stat().st_mtime) / 60
                if age_minutes < OPENREVIEW_COOKIE_MAX_AGE_MINUTES:
                    raw_cookies = json.loads(OPENREVIEW_COOKIE_PATH.read_text())
                    session_cookies = {c["name"]: c["value"] for c in raw_cookies}
                    log.debug(
                        f"OpenReview PDF: using {len(session_cookies)} session cookies "
                        f"({age_minutes:.1f}m old)"
                    )
                else:
                    log.warning(
                        f"OpenReview PDF: cookies are {age_minutes:.0f}m old — "
                        "download may be blocked"
                    )

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/pdf,*/*",
                "Referer": url,
                "Accept-Language": "en-US,en;q=0.9",
            }

            resp = requests.get(
                pdf_url,
                headers=headers,
                cookies=session_cookies,
                timeout=30,
                allow_redirects=True,
            )

            content_type = resp.headers.get("content-type", "")
            log.debug(
                f"OpenReview PDF response: HTTP {resp.status_code} "
                f"ct={content_type} len={len(resp.content)}"
            )

            if resp.status_code != 200:
                log.warning(
                    f"OpenReview PDF: HTTP {resp.status_code} — "
                    "cookies may be stale, re-run link extraction to refresh"
                )
                return ""

            if "pdf" not in content_type.lower() or len(resp.content) < 1000:
                log.warning(
                    f"OpenReview PDF: response doesn't look like a PDF "
                    f"(ct={content_type}, len={len(resp.content)})"
                )
                return ""

            pdf_bytes = resp.content
            log.debug(f"OpenReview PDF: downloaded {len(pdf_bytes)} bytes")

            text = extract_text(io.BytesIO(pdf_bytes), page_numbers=[0], maxpages=1)

            if not text or len(text) < 50:
                log.warning("OpenReview PDF: page 1 extraction returned too little text")
                return ""

            log.info(f"OpenReview PDF: extracted {len(text)} chars from page 1")
            return f"[OpenReview PDF page 1]\n{text[:8000]}"

        except ImportError:
            log.warning("pdfminer.six not installed — run: pip install pdfminer.six")
            return ""
        except Exception as e:
            log.warning(f"OpenReview PDF extraction failed: {e}")
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