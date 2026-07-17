# run_selenium_filler.py
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Import the main processing function from our other file
from selenium_filler import process_papers_with_selenium

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make sure `utils` (which lives at REPO_ROOT/utils) is importable regardless
# of the current working directory this script is launched from.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.ikdd_dedup import IKDDDeduplicator, refresh_cache, load_cache

# === ⚙️ CONFIGURATION ===
# EDIT THESE VALUES FOR EACH RUN
# ---
FORM_URL = "https://ikdd.hosting.acm.org/ds-papers-form.php"
CONFERENCE_NAME = "IEEE-ICDM"
YEAR = "2025"
MONTH = "Nov"   # must match the form dropdown exactly
VENUE = "ICDM"  # must match the form dropdown exactly

# If True, re-scrape IKDD for the latest approved papers before every run.
# If False, use whatever is already in the local cache (faster, but may be stale).
REFRESH_DEDUP_CACHE = True
# ---


def get_dedup_cache():
    """
    Returns a fresh (or cached) IKDD approved-titles cache, with a sensible
    fallback if a live refresh isn't possible.
    """
    if REFRESH_DEDUP_CACHE:
        try:
            print("🔍 Refreshing IKDD approved-papers cache (scraping latest list)...")
            return refresh_cache()
        except Exception as e:
            print(f"⚠️  Could not refresh IKDD cache live: {e}")
            print("   Falling back to local cache, if one exists...")

    try:
        return load_cache()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return None


def main():
    """
    Main function to find the data file, dedup-check the papers against
    IKDD's already-approved list, and start the Selenium form-filling
    process only for the papers that are genuinely new.
    """
    print("--- Starting Selenium Form Filling Process ---")

    # 1. Construct the path to the JSON file based on the config
    json_path = REPO_ROOT / f"data/final_output/{CONFERENCE_NAME}/{YEAR}/indian_papers_structured.json"

    # 2. Check if the file exists
    if not json_path.exists():
        print(f"❌ Error: Could not find the JSON file.")
        print(f"   Checked path: {json_path.resolve()}")
        return

    print(f"✅ Found data file: {json_path}")

    # 3. Load the paper data from the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            papers_data = json.load(f)
        except json.JSONDecodeError:
            print(f"❌ Error: The file at {json_path} is not a valid JSON file.")
            return

    if not papers_data:
        print("ℹ️ The JSON file is empty. Nothing to process.")
        return

    # 4. Dedup step — check candidate papers against IKDD's approved list
    #    BEFORE we open a browser or touch the form at all.
    cache = get_dedup_cache()
    if cache is None:
        print("❌ Aborting: no IKDD approved-titles cache available (live refresh "
              "failed and no local cache exists). Run with --refresh in "
              "utils/ikdd_dedup.py first, or fix IKDD_USERNAME/IKDD_PASSWORD.")
        return

    dedup = IKDDDeduplicator(cache=cache)
    new_papers, duplicates = dedup.filter_new(papers_data)

    print(f"\n📊 Dedup results: {len(papers_data)} total | "
          f"{len(new_papers)} new | {len(duplicates)} already in IKDD")

    if duplicates:
        print("   Skipping the following (already present in IKDD):")
        for p in duplicates:
            print(f"     [{p.get('_dedup_score', 0):.2f}] {p.get('paper_title', '')}"
                  f" → matched '{p.get('_dedup_matched', '')}'")

    if not new_papers:
        print("\n✅ Nothing to submit — every candidate paper is already in IKDD.")
        return

    # 5. Prepare the static configuration for the form
    form_config = {
        "form_url": FORM_URL,
        "venue": VENUE,
        "year": YEAR,
        "month": MONTH
    }

    # 6. Call the main processing function from our selenium_filler module,
    #    passing ONLY the papers that passed the dedup check.
    print(f"\n🚀 Proceeding to fill and submit {len(new_papers)} new paper(s)...")
    process_papers_with_selenium(new_papers, form_config)


if __name__ == "__main__":
    main()
