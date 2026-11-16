"""
Tier 1: Direct HTML scraper using requests + BeautifulSoup.
"""
import requests
from bs4 import BeautifulSoup
from .base import BaseScraper, ScrapeResult
from .challenge import looks_like_challenge


# AAAI's OJS host increasingly serves a JavaScript bot check to direct HTTP
# clients. Force those URLs to the visible browser tier instead of handing the
# interstitial to the extraction LLM.
BROWSER_REQUIRED_DOMAINS = ("ojs.aaai.org",)

# Sites known to work with direct HTML scraping.
HTML_FRIENDLY_DOMAINS = [
    "aclanthology.org",
    "proceedings.mlr.press",
    "aaai.org",
    "ijcai.org",
    "openaccess.thecvf.com",
    "vldb.org",
    "cidrdb.org",
    "usenix.org",
    "arxiv.org",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


class HTMLScraper(BaseScraper):

    def can_handle(self, url: str) -> bool:
        return any(domain in url for domain in HTML_FRIENDLY_DOMAINS)

    def scrape(self, url: str) -> ScrapeResult:
        try:
            if any(domain in url.lower() for domain in BROWSER_REQUIRED_DOMAINS):
                return ScrapeResult(
                    content="", source="html", url=url, success=False,
                    error="Browser rendering required for this site",
                )

            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()

            final_url = resp.url
            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type or final_url.endswith(".pdf"):
                return ScrapeResult(
                    content="", source="html", url=final_url,
                    success=False, error="Redirected to PDF"
                )

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "nav", "footer",
                             "header", "aside", "noscript"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # some bot-challenge pages (Cloudflare-style)
    
            text_lower = text.lower()
            if looks_like_challenge(text_lower, soup.title.get_text(" ", strip=True) if soup.title else ""):
                return ScrapeResult(
                    content="", source="html", url=final_url,
                    success=False, error="Bot challenge page detected — not real content"
                )

            if len(text) < 200:
                return ScrapeResult(
                    content="", source="html", url=final_url,
                    success=False, error="Page content too short — likely a login wall"
                )

            return ScrapeResult(
                content=text[:15000],
                source="html",
                url=final_url,
                success=True
            )

        except Exception as e:
            return ScrapeResult(
                content="", source="html", url=url,
                success=False, error=str(e)
            )
