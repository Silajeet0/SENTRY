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


@dataclass
class AffiliationDecision:
    label: str
    matches: list[str] = field(default_factory=list)


def classify_affiliation(affiliation: str) -> AffiliationDecision:
    text = f" {affiliation.lower()} "
    matches = []

    for pattern in EXPLICIT_POSITIVE_PATTERNS:
        if re.search(pattern, text):
            matches.append(pattern)

    for pattern in KNOWN_INDIAN_INSTITUTION_PATTERNS:
        if re.search(pattern, text):
            matches.append(pattern)

    if matches:
        return AffiliationDecision(label="positive", matches=matches)

    ambiguous = [pattern for pattern in AMBIGUOUS_PATTERNS if re.search(pattern, text)]
    if ambiguous:
        return AffiliationDecision(label="ambiguous", matches=ambiguous)

    return AffiliationDecision(label="negative", matches=[])


def combine_author_decisions(decisions: list[AffiliationDecision]) -> str:
    labels = {decision.label for decision in decisions}
    if "positive" in labels:
        return "positive"
    if "ambiguous" in labels:
        return "ambiguous"
    return "negative"
