# run_selenium_filler.py
import json
from pathlib import Path
# Import the main processing function from our other file
from selenium_filler import process_papers_with_selenium 

REPO_ROOT = Path(__file__).resolve().parents[1]

# === ⚙️ CONFIGURATION ===
# EDIT THESE VALUES FOR EACH RUN
# ---
FORM_URL = "https://ikdd.hosting.acm.org/ds-papers-form.php"
CONFERENCE_NAME = "IEEE-ICDM"
YEAR = "2025"
MONTH = "Nov"   # must match the form dropdown exactly
VENUE = "ICDM"  # must match the form dropdown exactly

# ---

def main():
    """
    Main function to find the data file and start the Selenium form-filling process.
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

    # 4. Prepare the static configuration for the form
    form_config = {
        "form_url": FORM_URL,
        "venue": VENUE,
        "year": YEAR,
        "month": MONTH
    }

    # 5. Call the main processing function from our selenium_filler module
    process_papers_with_selenium(papers_data, form_config)


if __name__ == "__main__":
    main()
