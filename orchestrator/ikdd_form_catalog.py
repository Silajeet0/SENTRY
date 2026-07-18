"""
ikdd_form_catalog.py — resolves a conference name to the exact "venue" and
"month" dropdown text the IKDD Premier Papers submission form
(ds-papers-form.php) expects.

Deliberately NOT LLM-guessed or derived, same philosophy as
conference_catalog.py's proceeding-URL resolution: Selenium's
Select.select_by_visible_text raises hard if the text doesn't match the
dropdown EXACTLY, and there's no way to derive the right text from the
conference name alone —

  - "IEEE-ICDM" needs "ICDM" on the form, not "IEEE-ICDM" — whatever
    string IKDD's admin actually typed into that <select> when they set
    the form up, which doesn't necessarily match AEGIS's own internal
    conference key.
  - "month" is whichever month the conference was actually held that
    year, not a fixed calendar slot — it can shift year to year.

So only conferences a human has actually verified against the live form
are listed here; everything else resolves=False and asks the person for
the exact text, the same way conference_catalog.resolve_conference_url
asks for an unknown ACM/IEEE URL rather than guessing one that might send
hours of scraping (or here, an hours-long Selenium run) down the wrong
path before anyone notices.

To add a new conference: open ds-papers-form.php, inspect the venue/month
<select> options for real, and add a verified line below.
"""
from orchestrator.conference_catalog import normalize_conference_name

# normalized_conference_key -> {"venue": <exact dropdown text>, "month": <exact dropdown text>}
# Carried over from values previously hand-set in Form_filler/run_selenium_filler.py's
# old hardcoded config for real runs of these conference/year pairs — not
# independently re-verified against the live form here, so double-check
# before trusting blindly, especially the month for a new year.
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
