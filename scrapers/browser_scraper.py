"""
Tier 2: Playwright headless browser scraper.
Handles JS-rendered pages that Tier 1 cannot.
Works for: IEEE Xplore
"""
import re
from pathlib import Path
from urllib.parse import urlparse
from .base import BaseScraper, ScrapeResult
from utils.ieee_author_parser import (
    extract_author_section,
    extract_document_id,
    format_author_affiliation_pairs,
)

# Domains that need JS rendering
BROWSER_DOMAINS = [
    "ieeexplore.ieee.org", #tested and worked
    "dl.acm.org",           
    "link.springer.com",
    "nature.com",
]

# Per-domain interaction rules: before reading the DOM, do these actions.
# Each entry is a list of (action, selector/value) tuples.
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

# if the access is blocked behind paywalls then we stop
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
        return any(domain in url for domain in BROWSER_DOMAINS) # check if the passed url is browser scraping friendly

    def scrape(self, url: str) -> ScrapeResult:
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) " # customize the HTTP header so that we aren't blocked
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                )
                page = context.new_page()

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000) # navigate to the desired page
                except PWTimeout:
                    browser.close()
                    return ScrapeResult(
                        content="", source="browser", url=url,
                        success=False, error="Page load timeout"
                    )

                # Run any domain-specific interactions
                domain = self._get_domain(url)
                for action, value in DOMAIN_INTERACTIONS.get(domain, []):
                    try:
                        # perform the actions
                        if action == "click":
                            page.click(value, timeout=5000)
                        elif action == "wait":
                            page.wait_for_timeout(int(value))
                        elif action == "scroll":
                            page.evaluate(f"window.scrollBy(0, {value})")
                    except Exception:
                        pass    # interaction failed — still try to read whatever is there

                # run JavaScript directly inside the browser's context to scrape specific <meta> tag where name
                # attribute starts with "citation"
                metadata = page.evaluate("""
                    () => Array.from(document.querySelectorAll('meta[name^="citation_"]'))
                        .map((meta) => {
                            const name = meta.getAttribute('name') || '';
                            const content = meta.getAttribute('content') || '';
                            return content ? `${name}: ${content}` : '';
                        })
                        .filter(Boolean)
                        .join('\\n')
                """)
                body_text = page.inner_text("body") # retrieves the visible text content of the entire <body> tag
                content = "\n".join(part for part in [metadata, body_text] if part) # merges the structured metadata and the body_text into one consolidated block of text

                # specific case for ieee since, ieee hide detailed author affiliations behind dynamic elements or 
                # separate sub-pages, a standard scrape of the main body text is usually insufficien
                if "ieeexplore.ieee.org/document/" in url:
                    authors_text = self._scrape_ieee_authors_page(page, url)
                    if authors_text:
                        content = (
                            "[IEEE authors and affiliations]\n"
                            + authors_text
                            + "\n\n[IEEE page metadata]\n"
                            + content
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
                content=content[:15000], # cap to avoid oversized LLM prompts
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

    @staticmethod
    def _get_domain(url: str) -> str:
        match = re.search(r"https?://([^/]+)", url) # regex to extract proper domain from url
        return match.group(1) if match else ""

    @staticmethod
    def _scrape_ieee_authors_page(page, url: str) -> str:
        """
        IEEE document pages can show only author names. The /authors#authors
        view exposes the author-affiliation pairs without requiring PDF access.
        """
        document_id = extract_document_id(url)
        if not document_id:
            return ""

        authors_url = f"https://ieeexplore.ieee.org/document/{document_id}/authors#authors" # specific url for the author affiliation
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

    @staticmethod
    def _debug_dump(url: str, content: str) -> None:
        debug_dir = Path("data/debug_scrapes")
        if not debug_dir.exists():
            return

        parsed = urlparse(url)
        document_match = re.search(r"/document/(\d+)", parsed.path)
        name = document_match.group(1) if document_match else re.sub(r"\W+", "_", url)[:80]
        path = debug_dir / f"{name}.txt"
        path.write_text(content, encoding="utf-8")
