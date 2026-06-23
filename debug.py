# test_acm_scraper.py
from playwright.sync_api import sync_playwright

TEST_URL = "https://dl.acm.org/doi/10.1145/3770854.3780159"

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled"]
    )
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    page = context.new_page()
    page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    """)

    # Step 1 — hit proceedings page first to clear Cloudflare
    # (same as the pipeline does during link extraction)
    print("Clearing Cloudflare via proceedings page...")
    page.goto(
        "https://dl.acm.org/doi/proceedings/10.1145/3770854",
        wait_until="domcontentloaded",
        timeout=60000
    )
    page.wait_for_function(
        "() => document.title.includes('Proceedings')",
        timeout=60000
    )
    print("Proceedings loaded — Cloudflare cleared")

    # Step 2 — now navigate to paper page (cookies already set)
    print(f"Loading paper: {TEST_URL}")
    page.goto(TEST_URL, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(5000)
    print("Paper title:", page.title())

    # Step 3 — click Authors Info & Claims
    for selector in [
        'a[data-id="article-authors-viewall"]',
        'text="Authors Info & Claims"',
    ]:
        try:
            loc = page.locator(selector).first
            if loc.count() == 0:
                continue
            loc.scroll_into_view_if_needed(timeout=3000)
            loc.click(timeout=5000)
            page.wait_for_function(
                "() => document.body.innerText.includes('Contributor Metrics')",
                timeout=15000
            )
            print("Authors panel opened")
            break
        except Exception as e:
            print(f"Selector failed: {selector} -> {e}")

    # Step 4 — extract affiliation text
    text = page.inner_text("body", timeout=5000)
    
    # Find the contributor section
    # Replace the existing print block with this:
    if "Contributor Metrics" in text:
        snippet = text[text.find("Contributor Metrics"):]
        
        for marker in ["Other Metrics", "Download PDF", "Footer", "About ACM"]:
            if marker in snippet:
                snippet = snippet[:snippet.find(marker)]
                break
        
        # Apply the same parser as browser_scraper.py
        lines = [l.strip() for l in snippet.splitlines() if l.strip()]
        skip = {"Contributor Metrics", "Expand All", "Collapse All"}
        lines = [l for l in lines if l not in skip]
        
        results = []
        i = 0
        while i < len(lines):
            line = lines[i]
            is_name = (
                len(line) < 60
                and "," not in line
                and not any(c.isdigit() for c in line)
                and line[0].isupper()
            )
            if is_name and i + 1 < len(lines):
                results.append(f"Author: {lines[i]} | Affiliation: {lines[i+1]}")
                i += 2
            else:
                i += 1
        
        print("\n=== FORMATTED OUTPUT ===")
        for r in results:
            print(r)
    else:
        print("Contributor Metrics not found")

    input("Press Enter to close browser...")
    browser.close()