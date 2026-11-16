"""Shared detection helpers for pages that have not reached their content yet."""
from __future__ import annotations


# Keep these phrases specific enough that normal paper text mentioning a
# security or research challenge is not mistaken for an anti-bot interstitial.
CHALLENGE_SIGNALS = (
    "checking your browser",
    "checking if the site connection is secure",
    "just a moment",
    "verify you are human",
    "verify that you are human",
    "human verification",
    "security verification",
    "performing security verification",
    "one moment while we verify",
    "access denied",
    "unusual traffic",
    "temporarily blocked",
    "challenge required",
    "ddos protection by",
    "enable javascript and cookies",
    "enable javascript to continue",
    "cf-chl-",
    "turnstile",
)


def looks_like_challenge(text: str = "", title: str = "") -> bool:
    haystack = f"{title}\n{text}".casefold()
    return any(signal in haystack for signal in CHALLENGE_SIGNALS)
