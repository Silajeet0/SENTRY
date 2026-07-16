import re
from dataclasses import dataclass, field


EXPLICIT_POSITIVE_PATTERNS = [
    r"\bindia\b",
    r"\bindian institute\b",
    r"\bindian statistical institute\b",
    r"\bindian institute of science\b",
    r"\bindian school of business\b",
]

KNOWN_INDIAN_INSTITUTION_PATTERNS = [
    r"\biit\b",
    r"\biiit\b",
    r"\bnit\b",
    r"\biisc\b",
    r"\bisi\b",
    r"\bbits pilani\b",
    r"\bjadavpur university\b",
    r"\banna university\b",
    r"\bamrita\b",
    r"\bmanipal\b",
    r"\bvit\b",
    r"\bdrdo\b",
    r"\bisro\b",
    r"\bcsir\b",
    r"\bbarc\b",
    r"\btifr\b",
    r"\biiser\b",
    r"\biim\b",
]

AMBIGUOUS_PATTERNS = [
    r"\btata\b",
    r"\binfosys\b",
    r"\bwipro\b",
    r"\bbengaluru\b",
    r"\bbangalore\b",
    r"\bhyderabad\b",
    r"\bmumbai\b",
    r"\bdelhi\b",
    r"\bchennai\b",
    r"\bkolkata\b",
    r"\bpune\b",
    r"\bahmedabad\b",
    r"\bjaipur\b",
]

# Explicit, unambiguous country names other than India. Used only to VETO a
# weak match (a short acronym like "isi", or a city name) when the SAME
# affiliation string also names a different country outright — e.g.
# "Pasteur Labs - ISI, United States" should not be "positive" just because
# it contains the substring "isi", when the string itself already says
# where the institution actually is. This does NOT override
# EXPLICIT_POSITIVE_PATTERNS — a literal "india"/"indian institute of X"
# match stays authoritative regardless (those aren't short/ambiguous
# tokens, so a stray co-occurring country name isn't grounds to doubt them).
NON_INDIA_COUNTRY_PATTERNS = [
    r"\bunited states\b", r"\busa\b",
    r"\bunited kingdom\b", r"\bengland\b", r"\bscotland\b",
    r"\bchina\b", r"\bgermany\b", r"\bfrance\b", r"\bjapan\b",
    r"\bsouth korea\b", r"\bcanada\b", r"\baustralia\b",
    r"\bsingapore\b", r"\bswitzerland\b", r"\bnetherlands\b",
    r"\bitaly\b", r"\bspain\b", r"\bisrael\b",
    r"\bhong kong\b", r"\btaiwan\b",
    r"\bunited arab emirates\b", r"\bsaudi arabia\b",
]


@dataclass
class AffiliationDecision:
    label: str
    matches: list[str] = field(default_factory=list)


def classify_affiliation(affiliation: str) -> AffiliationDecision:
    text = f" {affiliation.lower()} "

    # Explicit, unambiguous Indian institution names are authoritative —
    # nothing overrides these, including a stray co-occurring country name.
    explicit_matches = [p for p in EXPLICIT_POSITIVE_PATTERNS if re.search(p, text)]
    if explicit_matches:
        return AffiliationDecision(label="positive", matches=explicit_matches)

    weak_matches = [p for p in KNOWN_INDIAN_INSTITUTION_PATTERNS if re.search(p, text)]
    ambiguous_matches = [p for p in AMBIGUOUS_PATTERNS if re.search(p, text)]

    if weak_matches or ambiguous_matches:
        # Short acronyms (e.g. "isi") and city names alone are weak signals
        # that collide with unrelated institutions elsewhere. If the same
        # string also names a different country explicitly, that's strong
        # contradicting evidence — for OpenReview's structured
        # "Institution, Country" strings in particular, the country comes
        # straight from the author's own profile data, not an inference.
        if any(re.search(p, text) for p in NON_INDIA_COUNTRY_PATTERNS):
            return AffiliationDecision(label="negative", matches=[])
        if weak_matches:
            return AffiliationDecision(label="positive", matches=weak_matches)
        return AffiliationDecision(label="ambiguous", matches=ambiguous_matches)

    return AffiliationDecision(label="negative", matches=[])


def combine_author_decisions(decisions: list[AffiliationDecision]) -> str:
    labels = {decision.label for decision in decisions}
    if "positive" in labels:
        return "positive"
    if "ambiguous" in labels:
        return "ambiguous"
    return "negative"
