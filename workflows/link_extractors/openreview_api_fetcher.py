"""
openreview_api_fetcher.py — pure openreview-py API access for OpenReview-
hosted conferences (ICLR, ICML, NeurIPS-on-OpenReview, etc).

Required environment variables:
    OPENREVIEW_USERNAME
    OPENREVIEW_PASSWORD

"""
import os
import re
import logging
from functools import lru_cache

import openreview

log = logging.getLogger(__name__)

BASE_URL = "https://api2.openreview.net"

# OpenReview profiles store country as a two-letter ISO 3166-1 alpha-2 code
# (confirmed empirically)
_COUNTRY_CODE_MAP = {
    "IN": "India",
    "CN": "China",
    "US": "United States",
    "GB": "United Kingdom",
    "DE": "Germany",
    "FR": "France",
    "JP": "Japan",
    "KR": "South Korea",
    "CA": "Canada",
    "AU": "Australia",
    "SG": "Singapore",
    "CH": "Switzerland",
    "NL": "Netherlands",
    "IT": "Italy",
    "ES": "Spain",
    "IL": "Israel",
    "HK": "Hong Kong",
    "TW": "Taiwan",
    "AE": "United Arab Emirates",
    "SA": "Saudi Arabia",
}


def _expand_country(code: str) -> str:
    if not code:
        return ""
    return _COUNTRY_CODE_MAP.get(code.upper(), code)


@lru_cache(maxsize=1)
def get_client() -> "openreview.api.OpenReviewClient":
    username = os.environ.get("OPENREVIEW_USERNAME")
    password = os.environ.get("OPENREVIEW_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD must be set in the "
            "environment. Anonymous access to api2.openreview.net returns "
            "ChallengeRequiredError — authentication is required."
        )
    return openreview.api.OpenReviewClient(
        baseurl=BASE_URL, username=username, password=password
    )


def _venue_is_wanted(venue_text: str, skip_keywords: list[str], include_only: list[str]) -> bool:
    """
    Same skip_keywords/include_only substring-matching semantics used
    elsewhere in this pipeline 
    """
    text_lower = venue_text.lower()
    if include_only:
        return any(kw.lower() in text_lower for kw in include_only)
    return not any(kw.lower() in text_lower for kw in skip_keywords)


def fetch_accepted_notes(
    venue_id: str,
    skip_venue_keywords: list[str] | None = None,
    include_only_venue_keywords: list[str] | None = None,
) -> list["openreview.Note"]:
    """
    Fetch every accepted paper Note for a venue.

    Filtering by content={'venueid': venue_id} is what limits results to
    accepted papers specifically — submissions still under review, or
    rejected/withdrawn ones, carry a different venueid
    """
    if skip_venue_keywords is None:
        skip_venue_keywords = ["Workshop", "Tutorial"]
    if include_only_venue_keywords is None:
        include_only_venue_keywords = []

    client = get_client()

    log.info(f"Fetching accepted notes for venueid='{venue_id}'")
    notes = client.get_all_notes(content={"venueid": venue_id})
    log.info(f"Got {len(notes)} notes with venueid='{venue_id}' before venue-text filtering")

    filtered = []
    skipped_venues = set()
    for note in notes:
        venue_text = note.content.get("venue", {}).get("value", "")
        if _venue_is_wanted(venue_text, skip_venue_keywords, include_only_venue_keywords):
            filtered.append(note)
        else:
            skipped_venues.add(venue_text)

    if skipped_venues:
        log.info(f"Skipped venue labels: {sorted(skipped_venues)}")
    log.info(f"{len(filtered)}/{len(notes)} notes kept after filtering")

    return filtered


def fetch_author_profiles(client, authorids: list[str]) -> dict[str, dict]:
    """
    Batch-fetch profiles for a list of tilde author IDs (e.g. '~Jane_Doe1').
    Returns {authorid: {"name": str, "institution": str, "country": str}}.
    """
    if not authorids:
        return {}

    profiles = openreview.tools.get_profiles(client, authorids)
    result: dict[str, dict] = {}

    for profile in profiles:
        history = profile.content.get("history", [])
        current = next((h for h in history if h.get("end") is None),
                        history[0] if history else None)
        inst_name = ""
        country = ""
        if current:
            inst = current.get("institution", {}) or {}
            inst_name = inst.get("name", "") or ""
            country = inst.get("country", "") or ""

        names = profile.content.get("names", [])
        display_name = names[0].get("fullname", "") if names else profile.id

        result[profile.id] = {
            "name": display_name,
            "institution": inst_name,
            "country": country,
        }

    missing = [aid for aid in authorids if aid not in result]
    if missing:
        log.debug(f"No profile found for {len(missing)} author id(s): {missing}")

    return result


def build_paper_record(note, profiles_by_id: dict[str, dict]) -> dict:
    """
    Assemble the plain-dict view of one paper used by openreview_pipeline.py:
    title, abstract, forum URL, and a zipped authors list where each entry
    has name + affiliation string ("Institution, Country") ready to feed
    straight into evaluation.india_rules.classify_affiliation.
    """
    content = note.content
    authors = content.get("authors", {}).get("value", [])
    authorids = content.get("authorids", {}).get("value", [])

    zipped_authors = []
    for i, authorid in enumerate(authorids):
        display_name = authors[i] if i < len(authors) else authorid
        prof = profiles_by_id.get(authorid)
        if prof:
            inst = prof["institution"]
            country = _expand_country(prof["country"])
            affiliation = ", ".join(p for p in [inst, country] if p) or "Unknown"
            name = prof["name"] or display_name
        else:
            affiliation = "Unknown"
            name = display_name
        zipped_authors.append({"name": name, "affiliation": affiliation})

    return {
        "paper_url": f"https://openreview.net/forum?id={note.forum}",
        "title": content.get("title", {}).get("value", ""),
        "abstract": content.get("abstract", {}).get("value", ""),
        "venue": content.get("venue", {}).get("value", ""),
        "authors": zipped_authors,
    }


def fetch_single_openreview_paper(url: str) -> dict:
    """
    Fetch one paper by its forum URL (e.g. https://openreview.net/forum?id=XXXX),
    for single-paper testing/debugging via run_single_paper.py. Same output
    shape as the entries fetch_openreview_papers() returns, so it can go
    straight into process_openreview_paper() unchanged.
    """
    match = re.search(r"[?&]id=([^&]+)", url)
    if not match:
        raise ValueError(
            f"Could not extract a forum/note id from '{url}' — expected an "
            "openreview.net/forum?id=XXXX style URL."
        )
    note_id = match.group(1)

    client = get_client()
    note = client.get_note(id=note_id)

    authorids = note.content.get("authorids", {}).get("value", [])
    profiles_by_id = fetch_author_profiles(client, authorids)

    return build_paper_record(note, profiles_by_id)


def fetch_openreview_papers(
    venue_id: str,
    skip_venue_keywords: list[str] | None = None,
    include_only_venue_keywords: list[str] | None = None,
) -> list[dict]:
    """
    End-to-end: fetch accepted notes, batch-fetch all unique authors'
    profiles once, and return fully assembled paper records (title,
    abstract, authors-with-affiliations)
    """
    client = get_client()
    notes = fetch_accepted_notes(venue_id, skip_venue_keywords, include_only_venue_keywords)

    all_authorids: set[str] = set()
    for note in notes:
        all_authorids.update(note.content.get("authorids", {}).get("value", []))

    log.info(f"Batch-fetching profiles for {len(all_authorids)} unique authors...")
    profiles_by_id = fetch_author_profiles(client, list(all_authorids))

    return [build_paper_record(note, profiles_by_id) for note in notes]
