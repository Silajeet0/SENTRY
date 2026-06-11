"""
Tier 1: Direct HTML scraper using requests + BeautifulSoup.
Fast, free, zero dependencies beyond standard libs.
Works for: ACL Anthology, NeurIPS, ICML, ICLR, AAAI, IJCAI, CVPR (openaccess),
           ECCV (openaccess), SIGMOD, VLDB, most open-access proceedings.
"""
import requests
from bs4 import BeautifulSoup
from .base import BaseScraper, ScrapeResult

# Sites known to work with direct HTML scraping
HTML_FRIENDLY_DOMAINS = [
    "aclanthology.org",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",       # ICML, AISTATS, UAI
    "openreview.net",              # ICLR, NeurIPS workshops
    "aaai.org",
    "ijcai.org",
    "openaccess.thecvf.com",       # CVPR, ICCV, ECCV
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
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()

            # If the server redirected us to a PDF, signal failure so
            # the pipeline falls through to the PDF tier.
            final_url = resp.url
            content_type = resp.headers.get("content-type", "")
            if "pdf" in content_type or final_url.endswith(".pdf"):
                return ScrapeResult(
                    content="", source="html", url=final_url,
                    success=False, error="Redirected to PDF"
                )

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove nav, footer, scripts, styles — keep signal, not noise
            for tag in soup(["script", "style", "nav", "footer",
                             "header", "aside", "noscript"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Sanity check: at least 200 chars of real content
            if len(text) < 200:
                return ScrapeResult(
                    content="", source="html", url=final_url,
                    success=False, error="Page content too short — likely a login wall"
                )

            return ScrapeResult(
                content=text[:15000],   # cap to avoid oversized LLM prompts
                source="html",
                url=final_url,
                success=True
            )

        except Exception as e:
            return ScrapeResult(
                content="", source="html", url=url,
                success=False, error=str(e)
            )
