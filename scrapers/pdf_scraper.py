"""
Tier 3: PDF text extraction.
Used when the paper is only available as a PDF and/or previous tier fails.

"""
import io
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urlparse
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
        return url.endswith(".pdf") or "arxiv.org/pdf" in url

    def scrape(self, url: str) -> ScrapeResult:
        try:
            acl_metadata = self._try_acl_anthology_page(url)

            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and not url.endswith(".pdf"):
                return ScrapeResult(
                    content="", source="pdf", url=url,
                    success=False, error=f"Not a PDF (content-type: {content_type})"
                )

            pdf_bytes = io.BytesIO(resp.content)

            pdf_content = self._extract_first_page_text(pdf_bytes, url)
            content = self._join_content_sections(acl_metadata, pdf_content)

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
    def _extract_first_page_text(pdf_bytes: io.BytesIO, url: str = "") -> str:
        """
        ACL PDFs often encode visually separated words as tightly positioned
        glyphs with no actual spaces in the text stream, so they need
        pdfplumber's layout-aware extraction. NeurIPS PDFs generally expose a
        cleaner logical reading order through pdfminer, especially for
        multi-column author/affiliation blocks, so keep pdfminer first there.
        """
        extractors = (
            (PDFScraper._extract_with_pdfplumber, PDFScraper._extract_with_pdfminer)
            if "aclanthology.org" in url
            else (PDFScraper._extract_with_pdfminer, PDFScraper._extract_with_pdfplumber)
        )

        last_error = None
        for extractor in extractors:
            pdf_bytes.seek(0)
            try:
                text = extractor(pdf_bytes)
                if text.strip():
                    return PDFScraper._clean_pdf_text(text)
            except Exception as e:
                last_error = e

        if isinstance(last_error, ImportError):
            raise ImportError("pdfplumber or pdfminer.six not installed")
        return ""

    @staticmethod
    def _extract_with_pdfplumber(pdf_bytes: io.BytesIO) -> str:
        pdf_bytes.seek(0)
        import pdfplumber

        with pdfplumber.open(pdf_bytes) as pdf:
            if not pdf.pages:
                return ""
            return pdf.pages[0].extract_text(
                layout=True,
                x_tolerance=1,
                y_tolerance=3,
            ) or ""

    @staticmethod
    def _extract_with_pdfminer(pdf_bytes: io.BytesIO) -> str:
        pdf_bytes.seek(0)
        from pdfminer.high_level import extract_text

        return extract_text(pdf_bytes, page_numbers=[0], maxpages=1)

    @staticmethod
    def _clean_pdf_text(text: str) -> str:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    @staticmethod
    def _join_content_sections(*sections: str) -> str:
        return "\n\n".join(section.strip() for section in sections if section and section.strip())

    @staticmethod
    def _acl_html_url(url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc != "aclanthology.org":
            return ""

        html_url = url
        if html_url.endswith(".pdf"):
            html_url = html_url[:-4]
        return html_url if html_url.endswith("/") else html_url + "/"

    @staticmethod
    def _try_acl_anthology_page(url: str) -> str:
        html_url = PDFScraper._acl_html_url(url)
        if not html_url:
            return ""

        try:
            resp = requests.get(html_url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            meta: dict[str, list[str]] = {}
            for tag in soup.find_all("meta"):
                name = tag.get("name") or tag.get("property")
                value = tag.get("content")
                if name and value:
                    meta.setdefault(name, []).append(value.strip())

            lines = []
            titles = meta.get("citation_title") or meta.get("og:title")
            if titles:
                lines.append(f"Title: {titles[0]}")

            authors = meta.get("citation_author", [])
            if authors:
                lines.append("Authors: " + ", ".join(authors))

            venue = (meta.get("citation_conference_title") or meta.get("citation_journal_title") or [])
            if venue:
                lines.append(f"Venue: {venue[0]}")

            year = meta.get("citation_publication_date", [])
            if year:
                lines.append(f"Publication date: {year[0]}")

            doi = meta.get("citation_doi", [])
            if doi:
                lines.append(f"DOI: {doi[0]}")

            abstract = PDFScraper._extract_acl_abstract(soup)
            if abstract:
                lines.extend(["", "Abstract:", abstract])

            return "\n".join(lines) if len("\n".join(lines)) >= 100 else ""
        except Exception:
            return ""

    @staticmethod
    def _extract_acl_abstract(soup: BeautifulSoup) -> str:
        abstract_block = soup.select_one(".acl-abstract")
        if abstract_block:
            heading = abstract_block.find(["h1", "h2", "h3", "h4", "h5", "h6"])
            if heading:
                heading.decompose()
            return re.sub(r"\s+", " ", abstract_block.get_text(" ", strip=True)).strip()

        abstract_header = soup.find(string=lambda s: s and s.strip().lower() == "abstract")
        if not abstract_header:
            return ""

        container = abstract_header.find_parent(["div", "section", "main"])
        if not container:
            return ""

        text = container.get_text(separator="\n", strip=True)
        parts = text.split("Abstract", 1)
        if len(parts) != 2:
            return ""

        abstract = parts[1]
        for marker in (
            "Anthology ID:",
            "Volume:",
            "Month:",
            "Year:",
            "Address:",
        ):
            abstract = abstract.split(marker, 1)[0]
        return re.sub(r"\s+", " ", abstract).strip()

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
