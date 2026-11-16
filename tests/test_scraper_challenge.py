import unittest
from types import SimpleNamespace

from scrapers.browser_scraper import BrowserScraper
from scrapers.challenge import looks_like_challenge
from scrapers.html_scraper import HTMLScraper


class ScraperChallengeTests(unittest.TestCase):
    def test_aaai_ojs_is_forced_to_browser_tier(self):
        result = HTMLScraper().scrape("https://ojs.aaai.org/index.php/AAAI/article/view/37146")
        self.assertFalse(result.success)
        self.assertIn("Browser rendering required", result.error)

    def test_challenge_text_is_detected(self):
        self.assertTrue(looks_like_challenge("Checking your browser before accessing the site"))
        self.assertFalse(looks_like_challenge("Title: Graph neural networks\nAbstract: We study retrieval."))

    def test_browser_inspects_visible_page_text(self):
        page = SimpleNamespace(
            title=lambda: "Just a moment...",
            locator=lambda *_: SimpleNamespace(inner_text=lambda **__: "Checking your browser"),
        )
        self.assertTrue(BrowserScraper._is_challenge_page(page))


if __name__ == "__main__":
    unittest.main()
