import json
from pathlib import Path
from playwright.sync_api import sync_playwright

cookies = json.loads(Path("data/acm_session_cookies.json").read_text())
url = "https://dl.acm.org/doi/10.1145/3718958.3750472"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    context.add_cookies(cookies)
    page = context.new_page()
    page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(5000)

    result = page.evaluate("""
        () => {
            // Check if affiliation divs are inside author spans
            const authorSpan = document.querySelector('span[property="author"][typeof="Person"]');
            const affInsideAuthor = authorSpan?.querySelectorAll('div[property="affiliation"]').length;
            
            // Check parent of first affiliation div
            const firstAff = document.querySelector('div[property="affiliation"]');
            const affParentChain = [];
            let node = firstAff?.parentElement;
            for (let i = 0; i < 6 && node; i++) {
                affParentChain.push(node.tagName + '.' + (node.className || '').slice(0, 40));
                node = node.parentElement;
            }
            
            // Get the full outer HTML of first author span
            const authorHTML = authorSpan?.outerHTML?.slice(0, 500);
            
            // Check the data-doi attribute pattern used on the page
            const firstAffDiv = document.querySelector('div[property="affiliation"]');
            const affOuterHTML = firstAffDiv?.outerHTML?.slice(0, 300);
            
            return {
                affInsideAuthor,
                affParentChain,
                authorHTML,
                affOuterHTML,
            };
        }
    """)
    
    for k, v in result.items():
        print(f"\n{k}:")
        print(v)
    
    browser.close()