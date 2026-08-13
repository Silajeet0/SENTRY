"""
acm_api_fetcher.py — Fetches paper links for ACM proceedings via Playwright.

Opens the ACM DL proceedings page in a non-headless browser (required to
pass Cloudflare Turnstile), navigates to each SESSION section via its
?tocHeading=headingN URL, extracts all paper DOI links, and saves session
cookies for reuse by browser_scraper.py.

Supported conferences (all A* under ACM):
    KDD, SIGMOD, SIGIR, SIGCOMM, STOC, FOCS, PODC, SOSP, OSDI,
    CCS, SIGGRAPH, CHI, PLDI, ASPLOS, ICSE, FSE, WWW, CSCW, UIST

"""

import re
import json
import logging
import concurrent.futures
from pathlib import Path
from playwright.sync_api import sync_playwright

log = logging.getLogger(__name__)

# Where session cookies are saved for browser_scraper.py to reuse
COOKIE_PATH = Path("data/acm_session_cookies.json")


def _run_isolated(fn, *args, **kwargs):
    """
    Run fn (which opens sync_playwright()) in a brand-new, dedicated thread
    and block for its result.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args, **kwargs).result()


CONFERENCE_SESSION_CONFIG: dict[str, dict] = {

    "ACM_KDD": {
        "skip_sessions": ["Workshop", "Tutorial"],
        "include_only": [],
    },

    "ACM_CCS": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial"],
        "include_only": [],
    },

    "ACM_SIGMOD": {
        "skip_sessions": ["Keynote", "Demo", "Panel", "Tutorial", "Workshop"],
        "include_only": [],
    },


    "ACM_SIGGRAPH": {
        "skip_sessions": ["Keynote", "Course", "Talk", "Panel", "Poster", "Workshop", "Tutorial"],
        "include_only": [],
    },

    "ACM_SIGCOMM": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel", "Poster"],
        "include_only": [],
    },

    "ACM_EC": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },

    "ACM_FSE": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },


    "ACM_ISCA": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },

    "ACM_MOBICOM": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },

    "ACM_PODC": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },


    "ACM_PODS": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },

    "ACM_SIGIR": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },

    "ACM_SOSP": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },

    "ACM_STOC": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },

    "ACM_UIST": {
        "skip_sessions": ["Keynote", "Workshop", "Tutorial", "Panel"],
        "include_only": [],
    },
}

# Fallback when conference is not in CONFERENCE_SESSION_CONFIG.
_DEFAULT_CONFIG: dict = {
    "skip_sessions": ["Keynote", "Tutorial", "Workshop", "Demo", "Panel", "Poster"],
    "include_only": [],
}


def _get_conference_config(conference: str) -> dict:
    """Return the session config for *conference*, falling back to defaults."""
    key = conference.upper()
    if key in CONFERENCE_SESSION_CONFIG:
        return CONFERENCE_SESSION_CONFIG[key]
    for k in CONFERENCE_SESSION_CONFIG:
        if key.startswith(k):
            return CONFERENCE_SESSION_CONFIG[k]
    log.warning(
        f"No session config found for '{conference}'. "
        "Using default (skipping Keynote/Tutorial/Workshop/Demo/Panel/Poster). "
        "Add an entry to CONFERENCE_SESSION_CONFIG to customise behaviour."
    )
    return _DEFAULT_CONFIG


def _session_is_wanted(heading_text: str, cfg: dict) -> bool:
    """
    Return True if a SESSION heading should be harvested.

    """
    text_lower = heading_text.lower()
    if cfg["include_only"]:
        return any(s.lower() in text_lower for s in cfg["include_only"])
    return not any(s.lower() in text_lower for s in cfg["skip_sessions"])


def _wait_for_page_ready(page) -> None:
    """
    Block until Cloudflare challenge is cleared and the real ACM DL
    proceedings page is loaded.

    """
    log.info("Waiting for Cloudflare challenge to clear (solve in browser if prompted)...")

    try:
        page.wait_for_function(
            """() => {
                const t = document.title.toLowerCase();
                return t.length > 5 && !t.includes('just a moment');
            }""",
            timeout=120000,
        )
    except Exception as e:
        raise RuntimeError(
            "Cloudflare challenge never cleared after 120s. This launches a "
            "real (non-headless) Chromium window, so it needs an actual "
            "connected display/WindowServer session to render and pass the "
            "Turnstile check — a plain `ssh` session on macOS does NOT give "
            "the child process access to the logged-in user's WindowServer, "
            "so the browser can open but never really render/be interactive, "
            "and the challenge hangs forever. If you're driving this over "
            "SSH on macOS, run it inside the logged-in GUI session's "
            "bootstrap context instead, e.g.:\n"
            "    ssh user@host 'launchctl asuser $(id -u user) "
            "sudo -u user /path/to/venv/bin/python run.py'\n"
            "or connect via Screen Sharing/VNC and run it from a real "
            "logged-in terminal there. (Original error: "
            f"{type(e).__name__}: {e})"
        ) from e

    try:
        page.wait_for_selector(
            "a.accordion-tabbed__control[href*='tocHeading']",
            state="attached",
            timeout=30000,
        )
    except Exception:
        log.warning(
            "Main-content accordion anchors not found after page load — "
            f"title: '{page.title()}'. "
            "Page may be flat (no collapsible sections) or still loading."
        )

    log.info(f"Proceedings page ready — title: {page.title()}")


def _extract_paper_links(page) -> list[dict]:
    """Return all DOI anchor elements on the current page as {text, href}."""
    return page.eval_on_selector_all(
        'a[href*="/doi/10.1145/"]',
        "els => els.map(e => ({text: e.innerText.trim(), href: e.href}))",
    )


def _filter_paper_links(raw: list[dict], seen: set) -> list[str]:
    """
    Filter raw anchor dicts to genuine paper DOI URLs, deduplicating
    against *seen* (updated in place).
    """
    result = []
    for item in raw:
        href = item["href"].strip()
        title = item["text"].strip()
        if not title:
            continue
        if href in seen:
            continue
        if "/doi/proceedings/" in href:
            continue
        if not re.search(r"/doi/10\.1145/\d+\.\d+$", href):
            continue
        seen.add(href)
        result.append(href)
    return result


def _fetch_conference_links(
    proceeding_url: str,
    conference: str,
) -> tuple[list[str], list[dict]]:
    """Thread-isolated entry point"""
    return _run_isolated(_fetch_conference_links_impl, proceeding_url, conference)


def _fetch_conference_links_impl(
    proceeding_url: str,
    conference: str,
) -> tuple[list[str], list[dict]]:
    """
    Open the ACM DL proceedings page (non-headless to pass Cloudflare),
    navigate to each wanted SESSION's ?tocHeading=headingN URL to load its
    papers, collect all DOI links, and save cookies.

    Returns:
        (paper_links, cookies) — sorted list of DOI URLs and session cookies.
    """
    cfg = _get_conference_config(conference)
    cookies: list[dict] = []
    seen: set[str] = set()
    all_paper_links: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        page.goto(proceeding_url, timeout=120000)
        _wait_for_page_ready(page)

        # ── Step 1: collect papers already visible on the landing page ───────
        # The first session is pre-expanded on load — capture it now.
        landing_links = _extract_paper_links(page)
        new_links = _filter_paper_links(landing_links, seen)
        if new_links:
            log.info(f"Landing page: {len(new_links)} paper links")
            all_paper_links.extend(new_links)

        # ── Step 2: discover all SESSION headings with tocHeading URLs ────────
        # Query only main-content accordion anchors — href contains "tocHeading" (MIGHT NEED UPDATES IF STRUCUTRE IS CHANGED).
        session_entries: list[dict] = page.evaluate("""
            () => [
                ...document.querySelectorAll(
                    'a.accordion-tabbed__control[href*="tocHeading"]'
                )
            ].map(el => ({
                text: el.innerText.trim(),
                href: el.href,
                expanded: el.getAttribute('aria-expanded')
            }))
        """)

        log.info(f"Found {len(session_entries)} SESSION headings in main content")

        wanted = [s for s in session_entries if _session_is_wanted(s["text"], cfg)]
        skipped = [s for s in session_entries if not _session_is_wanted(s["text"], cfg)]

        log.info(f"Wanting {len(wanted)} sessions, skipping {len(skipped)}")
        for s in skipped:
            log.debug(f"  SKIP: {s['text']}")

        # ── Step 3: navigate to each wanted session's tocHeading URL ─────────
        # Each navigation is a full page load with that section pre-expanded.
        # We skip sections already expanded on the landing page (aria-expanded
        # = "true") since their links were already captured in Step 1.
        for entry in wanted:
            if entry["expanded"] == "true":
                log.debug(f"  already expanded on landing: {entry['text']}")
                continue

            log.info(f"  Loading: {entry['text']}")
            page.goto(entry["href"], wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(2000)

            raw = _extract_paper_links(page)
            new_links = _filter_paper_links(raw, seen)
            log.info(f"    → {len(new_links)} new paper links")
            all_paper_links.extend(new_links)

        # ── Step 4: save cookies after all navigation ─────────────────────────
        # Cloudflare clearance cookies persist across same-domain navigations.
        cookies = context.cookies()
        browser.close()

    log.info(f"Extracted {len(all_paper_links)} paper links total")
    return sorted(all_paper_links), cookies


def fetch_acm_links(
    proceeding_url: str,
    conference: str,
    year: str,
) -> str:
    """
    Fetch paper links for an ACM proceedings and save as grouped_links.json.
    Also saves session cookies for browser_scraper.py.

    Args:
        proceeding_url: Full ACM DL proceedings URL.
        conference:     Conference ID, e.g. "ACM_KDD", "ACM_CCS", "ACM_SIGCOMM".
        year:           Conference year string, e.g. "2025".

    Returns:
        Absolute path to the saved grouped_links.json file.
    """
    log.info(f"Fetching ACM proceedings for {conference} {year} via Playwright")

    paper_links, cookies = _fetch_conference_links(proceeding_url, conference)

    COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_PATH.write_text(json.dumps(cookies), encoding="utf-8")
    log.info(f"Saved {len(cookies)} ACM session cookies → {COOKIE_PATH}")

    if not paper_links:
        cfg = _get_conference_config(conference)
        raise RuntimeError(
            f"No paper links extracted from {proceeding_url}.\n"
            f"Conference config used: include_only={cfg['include_only']}, "
            f"skip_sessions={cfg['skip_sessions']}.\n"
            "Check that the expected SESSION headings exist on the page, "
            "or update CONFERENCE_SESSION_CONFIG for this conference."
        )

    cfg = _get_conference_config(conference)
    track_label = (
        ", ".join(cfg["include_only"]) if cfg["include_only"] else "All Sessions"
    )
    grouped_data = [
        {
            "track_title": track_label,
            "track_url": proceeding_url,
            "paper_links": paper_links,
        }
    ]

    save_dir = Path(f"data/links_raw/{conference}/{year}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "grouped_links.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Extracted {len(paper_links)} papers for {conference} {year}.")
    print(f"[INFO] Saved grouped links to: {save_path.resolve()}")

    return str(save_path.resolve())


def warmup_acm_cookies(proceeding_url: str) -> None:
    """Thread-isolated entry point — see _run_isolated() for why."""
    _run_isolated(_warmup_acm_cookies_impl, proceeding_url)


def _warmup_acm_cookies_impl(proceeding_url: str) -> None:
    """
    Visit the ACM proceedings page to clear Cloudflare and save session
    cookies.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        page.goto(proceeding_url, timeout=60000)

        log.info("Warmup: waiting for Cloudflare challenge to clear...")
        page.wait_for_function(
            """() => {
                const t = document.title.toLowerCase();
                return t.length > 5 && !t.includes('just a moment');
            }""",
            timeout=120000,
        )
        log.info(f"ACM warmup complete: {page.title()}")

        cookies = context.cookies()
        browser.close()

    COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_PATH.write_text(json.dumps(cookies), encoding="utf-8")
    log.info(f"Saved {len(cookies)} ACM session cookies → {COOKIE_PATH}")