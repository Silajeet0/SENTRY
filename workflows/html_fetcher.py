import time
from pathlib import Path


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def fetch_and_save_html(url: str, conference: str, year: str, wait: int = 5) -> str:
    print(f"[INFO] Fetching dynamic HTML from: {url}")

    html_content = _fetch_dynamic_html(url, conference, wait)

    with open("dynamic_page.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    save_dir = Path(f"data/html/{conference}/{year}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "proceedings.html"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[INFO] HTML saved to: {save_path.resolve()}")
    return str(save_path.resolve())


def _fetch_dynamic_html(url: str, conference: str, wait: int) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "playwright is required for dynamic proceedings pages. "
            "Install with: pip install playwright && playwright install chromium"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(wait * 1000)

        if "ieee" in conference.lower():
            html_content = _collect_ieee_paginated_html(page, wait)
        else:
            html_content = page.content()

        browser.close()
        return html_content


def _collect_ieee_paginated_html(page, wait: int) -> str:
    """
    IEEE Xplore proceedings are Angular pages with client-side pagination.
    The initial DOM only contains page 1, so collect every numbered page source
    and concatenate them before link extraction.
    """
    page_sources = []
    seen_page_numbers = set()

    while True:
        page.wait_for_selector("a[href*='/document/'], button[aria-label*='Page']", timeout=30000)
        page_sources.append(page.content())

        page_numbers = _ieee_page_numbers(page)
        current_page = _active_ieee_page(page)
        seen_page_numbers.add(current_page)

        next_pages = [num for num in page_numbers if num > current_page and num not in seen_page_numbers]
        if not next_pages:
            break

        next_page = next_pages[0]
        first_document = _first_ieee_document_href(page)
        selector = f"button[aria-label='Page {next_page} of search results']"
        print(f"[INFO] Fetching IEEE proceedings page {next_page}")

        button = page.locator(selector).first
        button.scroll_into_view_if_needed(timeout=5000)
        button.click(timeout=5000)

        page.wait_for_function(
            "(pageNumber) => document.querySelector('button[aria-label^=\"Page \"].active')?.textContent?.trim() === String(pageNumber)",
            arg=next_page,
            timeout=10000,
        )
        if first_document:
            page.wait_for_function(
                "(href) => document.querySelector('a[href*=\"/document/\"]')?.href !== href",
                arg=first_document,
                timeout=10000,
            )
        page.wait_for_timeout(1000)
        time.sleep(min(wait, 1))

    return "\n<!-- AEGIS_IEEE_PAGE_BREAK -->\n".join(page_sources)


def _ieee_page_numbers(page) -> list[int]:
    buttons = page.locator("button[aria-label^='Page ']")
    page_numbers = []
    for i in range(buttons.count()):
        button = buttons.nth(i)
        label = button.get_attribute("aria-label") or ""
        if "search results" not in label:
            continue
        text = (button.inner_text() or "").strip()
        if text.isdigit():
            page_numbers.append(int(text))
    return sorted(set(page_numbers))


def _active_ieee_page(page) -> int:
    active = page.locator("button[aria-label^='Page '].active")
    if active.count() == 0:
        return 1
    text = (active.first.inner_text() or "").strip()
    return int(text) if text.isdigit() else 1


def _first_ieee_document_href(page) -> str:
    links = page.locator("a[href*='/document/']")
    if links.count() == 0:
        return ""
    return links.first.get_attribute("href") or ""
