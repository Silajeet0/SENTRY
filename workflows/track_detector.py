from workflows.link_extractors.flat_link_extractor import extract_flat_links_with_base
from workflows.link_extractors.grouped_link_extractor import extract_grouped_links
from urllib.parse import urlparse

def is_track_grouped(conference: str, proceeding_url: str = "") -> bool:
    """
    Determines if the conference has papers grouped by tracks or sessions.
    Flat → IEEE, NeurIPS variants | Grouped → ACL, ACM, AAAI variants
    """
    domain = urlparse(proceeding_url or "").netloc.lower()
    if domain == "aclanthology.org":
        return True
    flat_keywords = ["ieee", "neurips", "nips", "icml"]
    grouped_keywords = ["acm", "acl", "aaai", "ijcai"]

    name = conference.lower()

    if any(key in name for key in flat_keywords):
        return False  # flat structure

    if any(key in name for key in grouped_keywords):
        return True  # grouped structure

    # Default behavior
    print(f"[WARN] Conference type unknown for: '{conference}' → assuming GROUPED.")
    return True



def extract_links_based_on_structure(conference: str, html_path: str, year: str, proceeding_url: str = "") -> list[str] | str:
    """
    Based on the structure (flat/grouped), uses the right extractor.
    """
    if is_track_grouped(conference, proceeding_url):
        return extract_grouped_links(html_path, conference, year, proceeding_url)
    else:
        return extract_flat_links_with_base(conference, html_path, year)