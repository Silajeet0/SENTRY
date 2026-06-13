"""
Base scraper interface. Every tier implements this.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ScrapeResult:
    content: str          # raw text content retrieved
    source: str           # which tier retrieved it e.g. "html", "browser", "pdf", "api"
    url: str              # final URL (may differ from input if redirected)
    success: bool         # retrieval successful or not indicator
    error: str = ""       # if error in retrieval then its reason


class BaseScraper(ABC):

    @abstractmethod
    def scrape(self, url: str) -> ScrapeResult:
        """
        Attempt to retrieve meaningful text content from url.
        Returns ScrapeResult with success=False if this tier cannot handle it.
        Never raises — errors go into ScrapeResult.error.
        """
        pass

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """
        Quick pre-check: should this tier even attempt this URL?
        Used to skip irrelevant tiers fast like skip PDF tier for HTML-only sites.
        """
        pass
