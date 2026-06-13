"""
Tier 4: academic API fallback.
Used when all browser/scraping tiers fail (paywalls, auth walls, etc.)

APIs used:
- Semantic Scholar Graph API  https://api.semanticscholar.org/graph/v1
- OpenAlex                    https://api.openalex.org
- CrossRef                    https://api.crossref.org

only limitation is that it needs a DOI or arXiv id to work
"""
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
from .base import BaseScraper, ScrapeResult

SS_BASE = "https://api.semanticscholar.org/graph/v1/paper"
OA_BASE = "https://api.openalex.org/works"
CR_BASE = "https://api.crossref.org/works"

HEADERS = {"User-Agent": "AEGIS-Pipeline/1.0 (academic research; contact@institution.edu)"}


def _extract_doi(url: str) -> str | None:
    """Extract DOI from common conference URL patterns."""
    patterns = [
        r"dl\.acm\.org/doi/(?:abs/|pdf/)?(10\.\d{4,}/[^?#]+)",    # ACM
        r"10\.\d{4,}/[^\s\"'<>?#]+",          # raw DOI anywhere in URL
        r"arxiv\.org/abs/([\d.]+)",          # arXiv
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            if "arxiv" in url: # if there is arXiv id in url use that instead
                continue
            doi = m.group(1) if m.lastindex else m.group(0)
            return _clean_doi(doi)
    return None


def _clean_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = unquote(doi).strip().rstrip(".,);]")
    return doi if doi.startswith("10.") and "/" in doi else None


def _extract_ieee_doi_from_page(url: str) -> str | None:
    """Resolve an IEEE Xplore document URL to its real DOI from page metadata."""
    if "ieeexplore.ieee.org" not in url:
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code >= 400:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        meta = soup.find("meta", attrs={"name": "citation_doi"})
        if meta and meta.get("content"):
            return _clean_doi(meta["content"])

        # IEEE also embeds JSON-ish metadata in scripts on many pages.
        match = re.search(r'"doi"\s*:\s*"(10\.\d{4,}/[^"]+)"', resp.text)
        if match:
            return _clean_doi(match.group(1))
    except Exception:
        return None

    return None


def _extract_arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\d.]+)", url)
    return m.group(1) if m else None


def _format_content(data: dict, source_api: str) -> str:
    """Turn API response into readable text the LLM extractor can parse."""
    lines = []
    title = data.get("title", "")
    if title:
        lines.append(f"Title: {title}")

    authors = data.get("authors", [])
    for a in authors:
        name = a.get("name", a.get("display_name", ""))
        affiliations = a.get("affiliations", [])
        if not affiliations:
            # OpenAlex nests differently
            institutions = a.get("institutions", [])
            affiliations = [i.get("display_name", "") for i in institutions]
        aff_str = "; ".join(affiliations) if affiliations else "Unknown"
        lines.append(f"Author: {name} | Affiliation: {aff_str}")

    lines.append(f"\n[Source: {source_api}]")
    return "\n".join(lines)


class APIScraper(BaseScraper):

    def can_handle(self, url: str) -> bool:
        # This tier is always a valid fallback — it tries regardless of domain
        return True

    def scrape(self, url: str) -> ScrapeResult:
        doi = _extract_doi(url) or _extract_ieee_doi_from_page(url)
        arxiv_id = _extract_arxiv_id(url)

        # Try Semantic Scholar
        ss_result = self._try_semantic_scholar(url, doi, arxiv_id)
        if ss_result.success:
            return ss_result

        # Try OpenAlex
        oa_result = self._try_openalex(url, doi, arxiv_id)
        if oa_result.success:
            return oa_result

        # Try CrossRef
        cr_result = self._try_crossref(doi)
        if cr_result.success:
            return cr_result

        return ScrapeResult(
            content="", source="api", url=url,
            success=False,
            error=f"All APIs failed. SS: {ss_result.error} | OA: {oa_result.error}"
        )

    # ------------------------------------------------------------------ #

    def _try_semantic_scholar(self, url, doi, arxiv_id) -> ScrapeResult:
        try:
            # Build lookup ID
            if doi:
                lookup = f"DOI:{doi}"
            elif arxiv_id:
                lookup = f"ARXIV:{arxiv_id}"
            else:
                return ScrapeResult(content="", source="semantic_scholar",
                                    url=url, success=False,
                                    error="No DOI or arXiv ID extractable")

            fields = "title,authors,externalIds"
            resp = requests.get(
                f"{SS_BASE}/{lookup}",
                params={"fields": fields},
                headers=HEADERS, timeout=15
            )
            if resp.status_code != 200:
                return ScrapeResult(content="", source="semantic_scholar",
                                    url=url, success=False,
                                    error=f"HTTP {resp.status_code}")

            paper = resp.json()
            paper_id = paper.get("paperId")
            if not paper_id:
                return ScrapeResult(content="", source="semantic_scholar",
                                    url=url, success=False, error="No paperId")

            # Second call to get full author affiliations
            authors_resp = requests.get(
                f"{SS_BASE}/{paper_id}/authors",
                params={"fields": "name,affiliations"},
                headers=HEADERS, timeout=15
            )
            if authors_resp.status_code == 200:
                paper["authors"] = authors_resp.json().get("data", [])

            content = _format_content(paper, "Semantic Scholar")
            return ScrapeResult(content=content, source="semantic_scholar",
                                url=url, success=True)

        except Exception as e:
            return ScrapeResult(content="", source="semantic_scholar",
                                url=url, success=False, error=str(e))

    def _try_openalex(self, url, doi, arxiv_id) -> ScrapeResult:
        try:
            if doi:
                lookup = f"https://doi.org/{doi}"
            elif arxiv_id:
                lookup = f"https://arxiv.org/abs/{arxiv_id}"
            else:
                return ScrapeResult(content="", source="openalex",
                                    url=url, success=False,
                                    error="No DOI or arXiv ID")

            resp = requests.get(
                OA_BASE,
                params={"filter": f"doi:{lookup}", "select": "title,authorships"},
                headers=HEADERS, timeout=15
            )
            if resp.status_code != 200:
                return ScrapeResult(content="", source="openalex",
                                    url=url, success=False,
                                    error=f"HTTP {resp.status_code}")

            results = resp.json().get("results", [])
            if not results:
                return ScrapeResult(content="", source="openalex",
                                    url=url, success=False, error="No results")

            work = results[0]
            # Normalise OpenAlex author structure to match our formatter
            authors = []
            for a in work.get("authorships", []):
                author_name = a.get("author", {}).get("display_name", "")
                institutions = [
                    i.get("display_name", "")
                    for i in a.get("institutions", [])
                ]
                authors.append({"name": author_name, "affiliations": institutions})
            work["authors"] = authors

            content = _format_content(work, "OpenAlex")
            return ScrapeResult(content=content, source="openalex",
                                url=url, success=True)

        except Exception as e:
            return ScrapeResult(content="", source="openalex",
                                url=url, success=False, error=str(e))

    def _try_crossref(self, doi) -> ScrapeResult:
        if not doi:
            return ScrapeResult(content="", source="crossref",
                                url="", success=False, error="No DOI")
        try:
            resp = requests.get(
                f"{CR_BASE}/{doi}",
                headers=HEADERS, timeout=15
            )
            if resp.status_code != 200:
                return ScrapeResult(content="", source="crossref",
                                    url="", success=False,
                                    error=f"HTTP {resp.status_code}")

            msg = resp.json().get("message", {})
            title = " ".join(msg.get("title", []))
            authors = []
            for a in msg.get("author", []):
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                aff = "; ".join(
                    x.get("name", "") for x in a.get("affiliation", [])
                )
                authors.append({"name": name, "affiliations": [aff] if aff else []})

            content = _format_content(
                {"title": title, "authors": authors}, "CrossRef"
            )
            return ScrapeResult(content=content, source="crossref",
                                url="", success=True)

        except Exception as e:
            return ScrapeResult(content="", source="crossref",
                                url="", success=False, error=str(e))
