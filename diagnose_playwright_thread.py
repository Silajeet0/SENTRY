"""
Standalone diagnostic — run this directly on the remote Mac:

    python diagnose_playwright_thread.py

It launches a persistent Playwright browser exactly the way BrowserScraper
does, uses it twice in a row from the same thread (should always work),
and prints diagnostic info about the Python/greenlet/playwright install.

If this script fails with the same "cannot switch to a different thread"
error even though everything here runs on one thread, that confirms the
problem is an environment/package issue (a broken or mismatched greenlet
build for this Python/arch) rather than anything in AEGIS's own code —
the fix at that point is reinstalling greenlet + playwright cleanly in
this venv, not further code changes.
"""
import sys
import platform
import threading

print(f"Python: {sys.version}")
print(f"Platform machine: {platform.machine()}")
print(f"Executable: {sys.executable}")
print(f"Main thread id (threading): {threading.get_ident()}")
print()

try:
    import greenlet
    print(f"greenlet version: {greenlet.__version__}")
    print(f"greenlet module file: {greenlet.__file__}")
except Exception as e:
    print(f"Could not import greenlet: {e}")

try:
    import playwright
    print(f"playwright module file: {playwright.__file__}")
    from importlib.metadata import version
    print(f"playwright version: {version('playwright')}")
except Exception as e:
    print(f"Could not import playwright: {e}")

print()
print(f"sys.path (checking for duplicate/shadowing installs):")
for p in sys.path:
    print(f"  {p}")

print()
print("--- Launching persistent browser (same pattern as BrowserScraper) ---")

from playwright.sync_api import sync_playwright

print(f"Thread id right before sync_playwright().start(): {threading.get_ident()}")
pw = sync_playwright().start()
print("sync_playwright().start() OK")

browser = pw.chromium.launch(headless=False, args=["--disable-blink-features=AutomationControlled"])
print("chromium.launch() OK")

context = browser.new_context(viewport={"width": 1920, "height": 1080})
print("new_context() OK")

page = context.new_page()
print(f"Thread id right before first page.goto(): {threading.get_ident()}")
page.goto("https://example.com", wait_until="domcontentloaded", timeout=30000)
print(f"First page.goto() OK — title: {page.title()!r}")
page.close()

# Reuse the SAME persistent browser/context for a second "paper" — this is
# the exact reuse pattern BrowserScraper relies on. Should always work
# since we're still on the same thread.
page2 = context.new_page()
print(f"Thread id right before second page.goto(): {threading.get_ident()}")
page2.goto("https://example.org", wait_until="domcontentloaded", timeout=30000)
print(f"Second page.goto() OK — title: {page2.title()!r}")
page2.close()

browser.close()
pw.stop()

print()
print("=== ALL STEPS SUCCEEDED — Playwright/greenlet install looks healthy. ===")
print("If this script passes but run_conference_summary.py still fails,")
print("something specific to the AEGIS import chain is involved — send")
print("this script's full output either way.")
