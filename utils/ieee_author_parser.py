import re
from urllib.parse import urlparse


SECTION_END_MARKERS = {
    "figures", "references", "citations", "keywords", "metrics", "footnotes",
    "document sections", "abstract",
}

STOP_LINES = {
    "view document", "publisher: ieee", "cite this", "pdf", "all authors",
    "authors", "subscribe", "donate", "cart", "create account", "personal sign in",
}

AFFILIATION_KEYWORDS = [
    "academy", "ai institute", "amazon", "association", "bank", "campus",
    "center", "centre", "college", "company", "corporation", "department",
    "google", "hospital", "ibm", "iit", "iiit", "inc.", "institute",
    "laboratories", "laboratory", "labs", "limited", "ltd", "meta",
    "microsoft", "national lab", "oracle", "research", "school", "tech",
    "technology", "univ", "university", "virginia tech",
]


def extract_document_id(url: str) -> str:
    match = re.search(r"ieeexplore\.ieee\.org/document/(\d+)", url)
    if match:
        return match.group(1)

    path_match = re.search(r"/document/(\d+)", urlparse(url).path)
    return path_match.group(1) if path_match else ""


def normalize_ieee_url(url: str) -> str:
    document_id = extract_document_id(url)
    return f"https://ieeexplore.ieee.org/document/{document_id}" if document_id else url


def extract_author_section(page_text: str) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return ""

    start = 0
    for i, line in enumerate(lines):
        if line.lower() == "authors":
            start = i + 1

    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].lower() in SECTION_END_MARKERS:
            end = i
            break

    return "\n".join(lines[start:end]).strip()


def looks_like_author_name(line: str) -> bool:
    if len(line) > 90 or "," in line or "|" in line:
        return False
    words = line.split()
    return 2 <= len(words) <= 7 and all(any(ch.isalpha() for ch in word) for word in words)


def looks_like_affiliation(line: str) -> bool:
    if not line or len(line) > 220:
        return False

    low = line.lower()
    if any(keyword in low for keyword in AFFILIATION_KEYWORDS):
        return True
    return not looks_like_author_name(line)


def extract_author_affiliation_pairs(author_section: str) -> list[dict]:
    lines = [line.strip(" ;") for line in author_section.splitlines() if line.strip(" ;")]
    cleaned = [
        line for line in lines
        if line.lower() not in STOP_LINES
        and not re.match(r"^\d+$", line)
        and not line.lower().startswith(("cites in", "full text views"))
    ]

    pairs = []
    i = 0
    while i < len(cleaned) - 1:
        name = cleaned[i]
        affiliation = cleaned[i + 1]
        if looks_like_author_name(name) and looks_like_affiliation(affiliation):
            pairs.append({"name": name, "affiliation": affiliation})
            i += 2
        else:
            i += 1

    return pairs


def format_author_affiliation_pairs(author_section: str) -> str:
    pairs = extract_author_affiliation_pairs(author_section)
    return "\n".join(
        f"Author: {pair['name']} | Affiliation: {pair['affiliation']}"
        for pair in pairs
    )
