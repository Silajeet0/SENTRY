import json
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List
import re
from urllib.parse import urljoin, urlparse
from workflows.link_extractors.skip_patterns import should_skip_title

# Base URLs mapped by keyword
BASE_URL_PATTERNS = {
    "ieee": {
        "base_url": "https://ieeexplore.ieee.org",
        "link_filter": lambda href, year: bool(
            re.match(r"^/(?:abstract/)?document/\d+/?$", urlparse(href).path)
        )
    },
    "neurips": {
        "base_url": "https://papers.nips.cc",
        "link_filter": lambda href, year: href.startswith(f"/paper_files/paper/{year}/hash/") and href.endswith(".html")
    },
    "nips": {
        "base_url": "https://papers.nips.cc",
        "link_filter": lambda href, year: href.startswith(f"/paper_files/paper/{year}/hash/") and href.endswith(".html")
    }
}


def extract_flat_links_with_base(conference: str, html_path: str, year: str) -> List[str]:
    """
    Extracts paper links from ungrouped (flat) HTML structure and saves them in JSON format.

    Args:
        conference (str): Conference ID (supports partial matching, e.g., ieee_icdm, nips_workshop)
        html_path (str): Path to the saved HTML file
        year (str): Year of the conference (used in dynamic path generation)

    Returns:
        List[str]: List of absolute paper links
    """

    conf_key = conference.lower()
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    all_anchors = soup.find_all("a", href=True)
    
    all_links = []
    for anchor in all_anchors:
        if should_skip_title(anchor.get_text(strip=True)):
            continue
        href = anchor.get("href", "").strip()
        if href:
            all_links.append(href)

    matched_config = None
    matched_keyword = None  # To identify which configuration was matched
    for keyword, config in BASE_URL_PATTERNS.items():
        if keyword in conf_key:
            matched_config = config
            matched_keyword = keyword
            break

    if not matched_config:
        raise ValueError(f"[ERROR] Unsupported flat conference pattern for: '{conference}'")

    base_url = matched_config["base_url"]
    link_filter = matched_config["link_filter"]
    paper_links = [link for link in all_links if link_filter(link, year)]

    # Process links, applying special modifications for neurips/nips
    processed_links = []
    for link in paper_links:
        full_link = base_url.rstrip("/")
        if matched_keyword in ["neurips", "nips"]:
            # For NeurIPS, transform the abstract link to the direct paper PDF link.
            # Example From: /paper_files/paper/2022/hash/...-Abstract-Conference.html
            # Example To:   /paper_files/paper/2022/file/...-Paper-Conference.pdf
            modified_link = link.replace("/hash/", "/file/")
            modified_link = modified_link.replace("Abstract", "Paper")
            modified_link = modified_link.replace(".html", ".pdf")
            full_link += modified_link
        elif matched_keyword == "ieee":
            path = urlparse(link).path
            document_match = re.search(r"/document/(\d+)/?$", path)
            if not document_match:
                continue
            full_link = f"{full_link}/document/{document_match.group(1)}"
        else:
            # For other conferences, just append the relative link
            full_link = urljoin(full_link + "/", link)
        processed_links.append(full_link)

    # Deduplicate and sort the final links
    full_links = sorted(list(set(processed_links)))

    # Save to dynamic output path
    save_dir = Path(f"data/links_raw/{conference}/{year}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "links.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(full_links, f, indent=4)

    print(f"[INFO] ✅ Extracted {len(full_links)} flat links")
    print(f"[INFO] 📁 Saved to: {save_path.resolve()}")
    return full_links
