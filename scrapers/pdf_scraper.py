"""
Tier 3: PDF text extraction.
Used when the paper is only available as a PDF (e.g. direct PDF links,
open-access CVF papers served as PDF, arXiv PDFs, etc.)

"""
import io
import requests
from bs4 import BeautifulSoup
from .base import BaseScraper, ScrapeResult

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


class PDFScraper(BaseScraper):

    def can_handle(self, url: str) -> bool:
        # Handle direct PDF links or arXiv
        return url.endswith(".pdf") or "arxiv.org/pdf" in url

    def scrape(self, url: str) -> ScrapeResult:
        try:
            from pdfminer.high_level import extract_text
        except ImportError:
            return ScrapeResult(
                content="", source="pdf", url=url,
                success=False, error="pdfminer.six not installed"
            )

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not url.endswith(".pdf"):
                return ScrapeResult(
                    content="", source="pdf", url=url,
                    success=False, error=f"Not a PDF (content-type: {content_type})"
                )

            pdf_bytes = io.BytesIO(resp.content)

            content = extract_text(pdf_bytes, page_numbers=[0], maxpages=1)

            if len(content) < 100:
                fallback_content = self._try_neurips_abstract_page(url)
                if fallback_content:
                    return ScrapeResult(
                        content=fallback_content,
                        source="neurips_html_fallback",
                        url=url,
                        success=True
                    )
                return ScrapeResult(
                    content="", source="pdf", url=url,
                    success=False, error="PDF text extraction returned too little content"
                )

            return ScrapeResult(
                content=content[:8000],
                source="pdf",
                url=url,
                success=True
            )

        except Exception as e:
            fallback_content = self._try_neurips_abstract_page(url)
            if fallback_content:
                return ScrapeResult(
                    content=fallback_content,
                    source="neurips_html_fallback",
                    url=url,
                    success=True
                )
            return ScrapeResult(
                content="", source="pdf", url=url,
                success=False, error=str(e)
            )

    @staticmethod
    def _try_neurips_abstract_page(url: str) -> str:
        if "papers.nips.cc" not in url or "/file/" not in url or not url.endswith(".pdf"):
            return ""

        abstract_url = url.replace("/file/", "/hash/").replace(".pdf", ".html")
        abstract_url = abstract_url.replace("-Paper-", "-Abstract-")

        try:
            resp = requests.get(abstract_url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text[:8000] if len(text) >= 100 else ""
        except Exception:
            return ""
