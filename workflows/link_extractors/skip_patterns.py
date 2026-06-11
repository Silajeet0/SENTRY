SKIP_TITLE_PATTERNS = [
    "chair message", "title page", "welcome message", "general chair",
    "program chair", "committee", "foreword", "in memoriam",
    "author index", "table of contents", "editor", "organizing committee"
]

def should_skip_title(title: str) -> bool:
    t = title.lower().strip()
    return any(pattern in t for pattern in SKIP_TITLE_PATTERNS)