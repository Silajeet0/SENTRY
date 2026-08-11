"""
ikdd_form_catalog.py — resolves a conference name to the exact "venue" and
"month" dropdown text the IKDD Premier Papers (or any other suitable venue) submission form
(ds-papers-form.php) expects.

To add a new conference: open ds-papers-form.php, (or corresponding suitable submissions page), inspect the venue/month
<select> options for real, and add a verified line below.
"""
from orchestrator.conference_catalog import normalize_conference_name

_KNOWN_FORM_METADATA: dict[str, dict] = {
    "IEEE-ICDM": {"venue": "ICDM", "month": "Nov"},
    "IEEE-CVPR": {"venue": "CVPR", "month": "Jun"},
}


def resolve(conference: str, year: str) -> dict:
    """
    Returns:
        {"conference", "year", "venue", "month", "resolved": True, ...}
    if a verified venue/month pair is on file, otherwise:
        {"conference", "year", "venue": None, "month": None,
         "resolved": False, "message": "..."}
    asking the person for the exact dropdown text instead of guessing one.
    """
    key = normalize_conference_name(conference)
    year = str(year).strip()
    meta = _KNOWN_FORM_METADATA.get(key)

    if meta:
        return {
            "conference": key,
            "year": year,
            "venue": meta["venue"],
            "month": meta["month"],
            "resolved": True,
            "method": "known",
            "notes": (
                f"Verified against the live form for a prior year — double "
                f"check '{meta['month']}' still matches {key} {year}'s actual "
                "month before relying on it, since conference dates can "
                "shift year to year."
            ),
        }

    return {
        "conference": key,
        "year": year,
        "venue": None,
        "month": None,
        "resolved": False,
        "method": "none",
        "message": (
            f"No verified venue/month dropdown text on file for {key}. Open "
            "the IKDD submission form (ds-papers-form.php), check the exact "
            "'venue' and 'month' dropdown options for this conference, and "
            "provide them directly — Selenium's select_by_visible_text fails "
            "hard on any mismatch, so guessing risks a failed run partway "
            "through rather than a clean, early error."
        ),
    }
