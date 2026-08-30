# selenium_filler.py
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def _is_others_area(selected_area: str) -> bool:
    return isinstance(selected_area, str) and selected_area.strip().lower() == "others"


def _fill_other_area_field(driver: webdriver.Chrome, other_text: str):
    """
    IKDD's (or any other suitable venue's) form reveals a follow-up free-text box directly below the Area
    of research dropdown the instant "Others" is selected 
    """
    text_to_enter = (other_text or "").strip()
    if not text_to_enter:
        text_to_enter = "Others"
        print("   ->   No specific 'Others' area text available from "
              f"extraction; falling back to '{text_to_enter}' in the "
              "follow-up box. Consider reviewing this paper's area "
              "classification manually after submission.")

    field = None
    try:
        field = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.NAME, "areaResearchOther"))
        )
    except TimeoutException:
        try:
            field = WebDriverWait(driver, 5).until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, "input[placeholder='Enter the area of research']")
                )
            )
        except TimeoutException:
            area_select = driver.find_element(By.NAME, "areaResearch")
            field = driver.execute_script(
                """
                const select = arguments[0];
                let container = select.closest('form') || select.parentElement;
                let node = select.parentElement;
                while (node && node !== container) {
                    const input = node.querySelector(
                        "input[type='text']:not([name='areaResearch'])"
                    );
                    if (input && input.offsetParent !== null) return input;
                    node = node.parentElement;
                }
                return null;
                """,
                area_select,
            )
            if field is None:
                raise NoSuchElementException(
                    "Could not locate the 'areaResearchOther' text field "
                    "that IKDD's form should reveal after selecting "
                    "'Others' — the form's markup may have changed."
                )

    driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", field)
    field.clear()
    field.send_keys(text_to_enter)


def fill_single_paper(driver: webdriver.Chrome, paper_data: dict, form_config: dict):
    """
    Fills the form for a single paper using an active browser instance.
    """
    try:
        # PHASE 1: PREPARE ALL AUTHOR FIELDS
        authors = paper_data.get('all_authors', [])
        num_authors = len(authors)
        clicks_needed = max(0, num_authors - 1)
        
        if clicks_needed > 0:
            add_author_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "addAuthor"))
            )
            for i in range(clicks_needed):
                print(f"   -> Clicking 'Add another author' ({i+1}/{clicks_needed})...")

                # Execute JavaScript to scroll the button into the center of the view.
                # This prevents other elements (like footers) from intercepting the click.
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", add_author_button)
                time.sleep(0.5) # A brief pause for the scroll to settle

                # Now that it's in view, wait for it to be clickable and then click
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "addAuthor"))
                ).click()
                
                print("      ...waiting for 10 seconds for page to update.")
                time.sleep(10)
        
        
        WebDriverWait(driver, 10).until(
            lambda d: len(d.find_elements(By.NAME, "authors[]")) == num_authors
        )

        # PHASE 2: FILL THE ENTIRE FORM
        print("   -> Filling paper title...")
        driver.find_element(By.NAME, "paperTitle").send_keys(paper_data['paper_title'])

        print("   -> Filling author details...")
        all_author_name_fields = driver.find_elements(By.NAME, "authors[]")
        all_affiliation_fields = driver.find_elements(By.NAME, "affiliations[]")
        for i, author in enumerate(authors):
            all_author_name_fields[i].send_keys(author['name'])
            all_affiliation_fields[i].send_keys(author['affiliation'])

        print("   -> Selecting dropdown options...")
        Select(driver.find_element(By.NAME, "venue")).select_by_visible_text(form_config['venue'])
        Select(driver.find_element(By.NAME, "year")).select_by_visible_text(form_config['year'])
        Select(driver.find_element(By.NAME, "month")).select_by_visible_text(form_config['month'])
        selected_area = paper_data['area_of_research']
        Select(driver.find_element(By.NAME, "areaResearch")).select_by_visible_text(selected_area)

        if _is_others_area(selected_area):
            print("   -> 'Others' selected — filling the follow-up area-of-research text box...")
            _fill_other_area_field(driver, paper_data.get('area_of_research_other', ''))

        # PHASE 3: SUBMIT
        print("   -> Submitting form...")
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@class='col-12 text-center btnAdd']/button"))
        )
        # Also scroll the submit button into view before clicking
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", submit_button)
        time.sleep(0.5)
        submit_button.click()
        
        # Wait for the success message to appear
        WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "success"))
        )
        print("   ->  Submission successful!")
        return True

    except (NoSuchElementException, TimeoutException) as e:
        print(f"\n   ->  An error occurred during form filling: Could not find or interact with an element.")
        print(f"      Details: {e}")
        return False
    except Exception as e:
        print(f"\n   ->  An unexpected error occurred: {e}")
        return False


def process_papers_with_selenium(papers: list, form_config: dict, on_progress=None) -> list:
    """
    Main processing function. Sets up a single browser instance and loops through
    all papers, filling the form for each one.

    on_progress, if given, is called as on_progress(results) after every
    single paper (submitted or failed) — so a caller (rpa_runner._job) can
    push live counts into RPA_REGISTRY as the run progresses, rather than
    the registry only updating once this whole loop finishes. This loop can
    legitimately run for well over an hour on ~80+ papers given the
    per-paper/per-author waits below, so incremental reporting is the
    difference between get_rpa_status being useful mid-run or not.

    Returns a list of per-paper result dicts:
        [{"paper_title": str, "status": "submitted" | "failed"}, ...]
    so callers (e.g. the orchestrator's RPA tool) can report real submission
    counts back to the person instead of only reading stdout.
    """
    results = []
    driver = None
    try:
        print(" Starting browser for the entire session...")
        service = webdriver.chrome.service.Service()
        driver = webdriver.Chrome(service=service)
        
        total_papers = len(papers)
        print(f"Found {total_papers} paper(s) to process.")

        for i, paper in enumerate(papers):
            title = paper.get('paper_title', 'No Title')
            print(f"\n--- Processing Paper {i+1}/{total_papers}: {title} ---")
            
            # Navigate to the form for each paper to start fresh
            driver.get(form_config["form_url"])
            
            # Call the function to fill the form for this specific paper
            success = fill_single_paper(driver, paper, form_config)
            results.append({"paper_title": title, "status": "submitted" if success else "failed"})

            if on_progress is not None:
                try:
                    on_progress(results)
                except Exception as e:  # noqa: BLE001
                    print(f"   -> (progress callback failed, continuing anyway: {e})")

            if success and i < total_papers - 1:
                print("--- Pausing for 5 seconds before next paper ---")
                time.sleep(5)
            elif not success:
                print("--- Pausing for 10 seconds after error before trying next paper ---")
                time.sleep(10)

        print("\n All papers have been processed.")

    finally:
        if driver:
            print("\nClosing browser...")
            driver.quit()

    return results
