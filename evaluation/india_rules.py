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
# Concretely: "IIT" collides with Istituto Italiano di Tecnologia (Italy),
# Illinois Institute of Technology (US), and Israel Institute of Technology
# (Technion is sometimes rendered this way) — this is exactly what produced
# the Peter Neri / "CHT Erzelli ... IIT ... Genova" false positive, where
# the affiliation is IIT Genova (Istituto Italiano di Tecnologia), not any
# Indian Institute of Technology. "ISI" collides with Istituto Superiore di
# Sanità (Italy) and Institute for Scientific Information; "NIT" collides
# with Nippon Institute of Technology (Japan); "VIT" is a common enough
# generic acronym elsewhere that it isn't safe alone either. These patterns
# never resolve to "positive" by themselves — see classify_affiliation.
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
# paired with one of these it's effectively unambiguous: no non-Indian "IIT"/
# "NIT"/"ISI"/"VIT" happens to also be named after Kanpur, Warangal, Vellore,
# etc. This is what lets "IIT Kanpur" or "NIT Rourkela" — real affiliations
# with no explicit "India" and no full spelled-out name — still resolve to
# positive, while a bare "IIT" (e.g. next to "Genova") does not.
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
    if weak_matches:
        # Even these low-collision names get vetoed if the string itself
        # names a different country outright — for OpenReview's structured
        # "Institution, Country" strings in particular, the country comes
        # straight from the author's own profile data, not an inference.
        if any(re.search(p, text) for p in NON_INDIA_COUNTRY_PATTERNS):
            return AffiliationDecision(label="negative", matches=[])
        return AffiliationDecision(label="positive", matches=weak_matches)

    acronym_matches = [p for p in AMBIGUOUS_ACRONYM_INSTITUTION_PATTERNS if re.search(p, text)]
    campus_matches = [p for p in INDIAN_ACRONYM_CAMPUS_PATTERNS if re.search(p, text)]
    ambiguous_matches = [p for p in AMBIGUOUS_PATTERNS if re.search(p, text)]

    has_foreign_country = any(re.search(p, text) for p in NON_INDIA_COUNTRY_PATTERNS)

    # Corroboration for a bare acronym can come from either list: a
    # campus-specific town (Kanpur, Warangal, Vellore, ...) or one of the
    # general Indian city/company names (e.g. "ISI Kolkata").
    corroboration = campus_matches + ambiguous_matches
    if acronym_matches and corroboration and not has_foreign_country:
        # e.g. "IIT Kanpur", "NIT Warangal", "ISI Kolkata", "VIT Vellore" —
        # the acronym paired with a real Indian town/city is effectively
        # unambiguous, even with no "India" stated.
        return AffiliationDecision(label="positive", matches=acronym_matches + corroboration)

    if acronym_matches or ambiguous_matches:
        # Collision-prone acronyms (IIT/NIT/ISI/VIT — each one is also a
        # real non-Indian institution's abbreviation somewhere in the
        # world) and bare city/company names are never, by themselves,
        # strong enough evidence to call an author Indian-affiliated. They
        # always resolve to "ambiguous" here, and every caller in this
        # codebase treats "ambiguous" the same as "negative" for anything
        # that ends up in production output — the label is kept distinct
        # only so ground-truth generation can bucket these for manual
        # review. There is deliberately no veto/override in the other
        # direction: this is a case where under-counting Indian papers is
        # the safer failure mode than incorrectly flagging a non-Indian
        # author (e.g. "IIT" = Istituto Italiano di Tecnologia).
        return AffiliationDecision(label="ambiguous", matches=acronym_matches + ambiguous_matches)

    return AffiliationDecision(label="negative", matches=[])


def combine_author_decisions(decisions: list[AffiliationDecision]) -> str:
    labels = {decision.label for decision in decisions}
    if "positive" in labels:
        return "positive"
    if "ambiguous" in labels:
        return "ambiguous"
    return "negative"
