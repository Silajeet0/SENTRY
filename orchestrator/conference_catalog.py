"""
conference_catalog.py — backs the resolve_conference_url, validate_url, and
detect_structure orchestrator tools.

Deliberately NOT LLM-guessed. A wrong proceedings URL doesn't fail loudly —
it sends the four-tier scraper cascade to scrape the wrong conference's
papers for hours before anyone notices. So every URL here is either:
  (a) a stable, predictable pattern (NeurIPS, ACL-family, OpenReview), or
  (b) a hand-verified DOI / conhome URL pulled from a prior real run (ACM,
      IEEE) — those are per-instance identifiers with no stable pattern
      across years, so unknown ones resolve=False and ask for the URL
      rather than fabricate one.
"""
from typing import Optional
from urllib.parse import urlparse

import requests

from main_driver import OPENREVIEW_CONFERENCES
from workflows.html_fetcher import USER_AGENT
from workflows.track_detector import is_track_grouped

# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------
_ALIASES = {
    "NEURIPS": "NeurIPS", "NIPS": "NeurIPS",
    "ICML": "ICML", "ICLR": "ICLR",
    "ACL": "ACL", "EMNLP": "EMNLP", "NAACL": "NAACL",
    "KDD": "ACM_KDD", "ACM-KDD": "ACM_KDD", "ACM KDD": "ACM_KDD", "ACM_KDD": "ACM_KDD",
    "SIGCOMM": "ACM_SIGCOMM", "ACM-SIGCOMM": "ACM_SIGCOMM", "ACM_SIGCOMM": "ACM_SIGCOMM",
    "CCS": "ACM_CCS", "ACM_CCS": "ACM_CCS",
    "SIGMOD": "ACM_SIGMOD", "ACM_SIGMOD": "ACM_SIGMOD",
    "SIGGRAPH": "ACM_SIGGRAPH", "ACM_SIGGRAPH": "ACM_SIGGRAPH",
    "SIGIR": "ACM_SIGIR", "ACM_SIGIR": "ACM_SIGIR",
    "SOSP": "ACM_SOSP", "ACM_SOSP": "ACM_SOSP",
    "STOC": "ACM_STOC", "ACM_STOC": "ACM_STOC",
    "UIST": "ACM_UIST", "ACM_UIST": "ACM_UIST",
    "PODC": "ACM_PODC", "ACM_PODC": "ACM_PODC",
    "PODS": "ACM_PODS", "ACM_PODS": "ACM_PODS",
    "EC": "ACM_EC", "ACM_EC": "ACM_EC",
    "FSE": "ACM_FSE", "ACM_FSE": "ACM_FSE",
    "ISCA": "ACM_ISCA", "ACM_ISCA": "ACM_ISCA",
    "MOBICOM": "ACM_MOBICOM", "ACM_MOBICOM": "ACM_MOBICOM",
    "ICDM": "IEEE-ICDM", "IEEE-ICDM": "IEEE-ICDM",
    "FOCS": "IEEE-FOCS", "IEEE-FOCS": "IEEE-FOCS",
    "HRI": "IEEE-HRI", "IEEE-HRI": "IEEE-HRI",
    "ICCV": "IEEE-ICCV", "IEEE-ICCV": "IEEE-ICCV",
    "ICRA": "IEEE-ICRA", "IEEE-ICRA": "IEEE-ICRA",
    "ISMAR": "IEEE-ISMAR", "IEEE-ISMAR": "IEEE-ISMAR",
    "LICS": "IEEE-LICS", "IEEE-LICS": "IEEE-LICS",
    "PERCOM": "IEEE-PERCOM", "IEEE-PERCOM": "IEEE-PERCOM",
    "IEEE-SP": "IEEE-SP",
    "IEEE-CV": "IEEE-CV",
    "CVPR": "IEEE-CVPR", "IEEE-CVPR": "IEEE-CVPR",
}

# Hand-verified proceeding URLs pulled from prior real runs (see run.py).
# ACM DOIs and IEEE conhome IDs are per-instance — there is no way to derive
# next year's from this year's, so only exactly these (conference, year)
# pairs resolve automatically.
_KNOWN_INSTANCES: dict[tuple[str, str], str] = {
    ("IEEE-ICDM", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/11391637/proceeding",
    ("IEEE-FOCS", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/11368763/proceeding",
    ("IEEE-HRI", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/10973274/proceeding",
    ("IEEE-ICCV", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/11443115/proceeding",
    ("IEEE-ICRA", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/11127273/proceeding",
    ("IEEE-ISMAR", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/11220258/proceeding",
    ("IEEE-LICS", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/11186120/proceeding",
    ("IEEE-PERCOM", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/11524112/proceeding",
    ("IEEE-SP", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/11023178/proceeding",
    ("IEEE-CV", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/10937339/proceeding",
    ("IEEE-CVPR", "2025"): "https://ieeexplore.ieee.org/xpl/conhome/11091818/proceeding",
    ("ACM_KDD", "2026v1"): "https://dl.acm.org/doi/proceedings/10.1145/3770854",
    ("ACM_SIGCOMM", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3718958",
    ("ACM_CCS", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3719027",
    ("ACM_EC", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3736252",
    ("ACM_FSE", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3696630",
    ("ACM_ISCA", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3695053",
    ("ACM_MOBICOM", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3680207",
    ("ACM_PODC", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3732772",
    ("ACM_PODS", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3722234",
    ("ACM_SIGGRAPH", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3721238",
    ("ACM_SIGIR", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3726302",
    ("ACM_SIGMOD", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3722212",
    ("ACM_SOSP", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3731569",
    ("ACM_STOC", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3717823",
    ("ACM_UIST", "2025"): "https://dl.acm.org/doi/proceedings/10.1145/3746058",
}


def _pattern_url(key: str, year: str) -> Optional[str]:
    """Stable, derivable-from-year-alone URL patterns."""
    if key == "NeurIPS":
        return f"https://papers.nips.cc/paper_files/paper/{year}"
    if key in {"ACL", "EMNLP", "NAACL"}:
        return f"https://aclanthology.org/events/{key.lower()}-{year}/"
    if key in OPENREVIEW_CONFERENCES:
        return f"https://openreview.net/group?id={key}.cc/{year}/Conference"
    return None


def resolve_conference_url(conference: str, year: str) -> dict:
    """Resolve a conference name + year to its proceedings URL."""
    raw = (conference or "").strip()
    key = _ALIASES.get(raw.upper(), raw)
    year = str(year).strip()

    known = _KNOWN_INSTANCES.get((key, year))
    if known:
        return {
            "conference": key,
            "year": year,
            "proceeding_url": known,
            "resolved": True,
            "method": "known_instance",
        }

    pattern_url = _pattern_url(key, year)
    if pattern_url:
        return {
            "conference": key,
            "year": year,
            "proceeding_url": pattern_url,
            "resolved": True,
            "method": "pattern",
        }

    return {
        "conference": key,
        "year": year,
        "proceeding_url": None,
        "resolved": False,
        "method": "none",
        "message": (
            f"No known or predictable proceedings URL for {key} {year}. "
            "ACM and IEEE proceeding URLs are per-instance DOIs/conhome IDs "
            "with no stable pattern across years — please provide the URL "
            "directly (e.g. https://dl.acm.org/doi/proceedings/10.1145/XXXXXXX "
            "or https://ieeexplore.ieee.org/xpl/conhome/XXXXXXXX/proceeding)."
        ),
    }


# Domains that challenge-wall plain HTTP requests but are scraped fine via
# the pipeline's own Playwright-based fetchers — a non-2xx here doesn't mean
# the URL is actually bad.
_CHALLENGE_WALLED_DOMAINS = {"dl.acm.org", "openreview.net", "ieeexplore.ieee.org"}


def validate_url(url: str) -> dict:
    """Lightweight reachability check before committing to a full run."""
    try:
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=15, allow_redirects=True
        )
        ok = resp.status_code < 400
        result = {
            "url": url,
            "status_code": resp.status_code,
            "ok": ok,
            "final_url": resp.url,
        }
        domain = urlparse(url).netloc
        if not ok and any(d in domain for d in _CHALLENGE_WALLED_DOMAINS):
            result["note"] = (
                "A non-2xx/3xx from a plain HTTP request doesn't necessarily "
                "mean this URL is bad — AEGIS fetches this domain with a real "
                "Playwright browser to get past its bot challenge, which "
                "behaves very differently from a bare request. Proceed with "
                "run_pipeline; the browser-based fetcher will report a "
                "clearer error if the URL is actually wrong."
            )
        return result
    except requests.RequestException as e:
        return {"url": url, "ok": False, "status_code": None, "error": str(e)}


def detect_structure(conference: str, proceeding_url: str = "") -> dict:
    """
    Determine which of AEGIS's three scraping paths a conference will take.
    Mirrors main_driver.run_pipeline's own branch selection exactly, so the
    orchestrator's mental model can't drift from what will actually run.
    """
    base = (conference or "").upper().split("_")[0]

    if base in OPENREVIEW_CONFERENCES:
        return {
            "conference": conference,
            "structure": "grouped",
            "handler": "openreview",
            "notes": (
                "Fetched via Playwright group-page load + clicking each "
                "Accept tab, then track selection."
            ),
        }

    if "dl.acm.org" in (proceeding_url or ""):
        return {
            "conference": conference,
            "structure": "grouped",
            "handler": "acm_dl",
            "notes": (
                "Fetched via Playwright session-heading navigation "
                "(Cloudflare bypass), then track selection."
            ),
        }

    grouped = is_track_grouped(conference)
    return {
        "conference": conference,
        "structure": "grouped" if grouped else "flat",
        "handler": "generic_grouped" if grouped else "generic_flat",
        "notes": (
            "Fetched via html_fetcher, then "
            + ("grouped_link_extractor + track selection." if grouped else "flat_link_extractor.")
        ),
    }
