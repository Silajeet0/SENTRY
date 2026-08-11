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
    r"\biiit\b",
    r"\biisc\b",
    r"\bbits pilani\b",
    r"\bjadavpur university\b",
    r"\banna university\b",
    r"\bamrita\b",
    r"\bmanipal\b",
    r"\bdrdo\b",
    r"\bisro\b",
    r"\bcsir\b",
    r"\bbarc\b",
    r"\btifr\b",
    r"\btata institute of fundamental research\b",
    r"\biiser\b",
    r"\biim\b",
]

# Short acronyms that are NOT reliable on their own: each one is also the
# real, in-use abbreviation for at least one non-Indian institution, so a
# bare match is genuinely ambiguous rather than weak-but-probably-Indian.

AMBIGUOUS_ACRONYM_INSTITUTION_PATTERNS = [
    r"\biit\b",
    r"\bnit\b",
    r"\bisi\b",
    r"\bvit\b",
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

# Towns/cities that IIT, NIT, ISI, and VIT campuses are actually named after
# (including older names still in common use, e.g. "Bombay"/"Madras"/
# "Calcutta"). A bare acronym match alone stays "ambiguous" (see below), but
# paired with one of these it's effectively unambiguous
INDIAN_ACRONYM_CAMPUS_PATTERNS = [
    r"\bkanpur\b", r"\bkharagpur\b", r"\bbombay\b", r"\bmadras\b", r"\bcalcutta\b",
    r"\bguwahati\b", r"\broorkee\b", r"\bropar\b", r"\bbhubaneswar\b",
    r"\bgandhinagar\b", r"\bjodhpur\b", r"\bpatna\b", r"\bindore\b", r"\bmandi\b",
    r"\bvaranasi\b", r"\bpalakkad\b", r"\btirupati\b", r"\bdhanbad\b",
    r"\bbhilai\b", r"\bdharwad\b", r"\bjammu\b", r"\bgoa\b",
    r"\bagartala\b", r"\bbhopal\b", r"\bcalicut\b", r"\bdurgapur\b",
    r"\bhamirpur\b", r"\bjalandhar\b", r"\bjamshedpur\b", r"\bkurukshetra\b",
    r"\bnagpur\b", r"\bpuducherry\b", r"\braipur\b", r"\brourkela\b",
    r"\bsilchar\b", r"\bsrinagar\b", r"\bsurathkal\b", r"\btiruchirapalli\b",
    r"\btrichy\b", r"\bwarangal\b", r"\bvellore\b", r"\bamaravati\b",
    r"\btezpur\b",
]

# Explicit, unambiguous country names other than India. Used only to VETO a
# weak match (a short acronym like "isi", or a city name) when the SAME
# affiliation string also names a different country outright
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
    explicit_matches = [p for p in EXPLICIT_POSITIVE_PATTERNS if re.search(p, text)]
    if explicit_matches:
        return AffiliationDecision(label="positive", matches=explicit_matches)

    weak_matches = [p for p in KNOWN_INDIAN_INSTITUTION_PATTERNS if re.search(p, text)]
    if weak_matches:
        # Even these low-collision names get vetoed if the string itself
        # names a different country outright
        if any(re.search(p, text) for p in NON_INDIA_COUNTRY_PATTERNS):
            return AffiliationDecision(label="negative", matches=[])
        return AffiliationDecision(label="positive", matches=weak_matches)

    acronym_matches = [p for p in AMBIGUOUS_ACRONYM_INSTITUTION_PATTERNS if re.search(p, text)]
    campus_matches = [p for p in INDIAN_ACRONYM_CAMPUS_PATTERNS if re.search(p, text)]
    ambiguous_matches = [p for p in AMBIGUOUS_PATTERNS if re.search(p, text)]

    has_foreign_country = any(re.search(p, text) for p in NON_INDIA_COUNTRY_PATTERNS)


    corroboration = campus_matches + ambiguous_matches
    if acronym_matches and corroboration and not has_foreign_country:

        return AffiliationDecision(label="positive", matches=acronym_matches + corroboration)

    if acronym_matches or ambiguous_matches:
        return AffiliationDecision(label="ambiguous", matches=acronym_matches + ambiguous_matches)

    return AffiliationDecision(label="negative", matches=[])


def combine_author_decisions(decisions: list[AffiliationDecision]) -> str:
    labels = {decision.label for decision in decisions}
    if "positive" in labels:
        return "positive"
    if "ambiguous" in labels:
        return "ambiguous"
    return "negative"
