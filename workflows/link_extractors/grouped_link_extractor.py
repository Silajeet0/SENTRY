import json
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from workflows.link_extractors.skip_patterns import should_skip_title

def extract_grouped_links(html_path: str, conference: str, year: str) -> str:
    """
    Extracts paper links grouped by tracks/sessions for ACL and ACM-style conferences.

    Supports:
    - ACL (aclanthology.org)
    - ACM variants (acm_kdd, acm_ikdd, acm_mod)

    Args:
        html_path (str): Path to the saved HTML file
        conference (str): Any conference ID (used to infer style)
        year (str): Conference year

    Returns:
        str: Path to the saved grouped links JSON
    """
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    grouped_data = []
    conf = conference.lower()

    # === ACL STYLE ===
    if "acl" in conf:
        base_url = "https://aclanthology.org"
        for track_anchor in soup.find_all("a", class_="align-middle", href=True):
            href = track_anchor["href"].strip()
            title = track_anchor.get_text(strip=True)

            if not href.startswith("/volumes/"):
                continue

            track = {
                "track_title": title,
                "track_url": base_url + href,
                "paper_links": []
            }

            current = track_anchor.find_parent("h4")
            if not current:
                continue

            sibling = current.find_next_sibling()
            while sibling and sibling.name != "h4":
                for pdf_link in sibling.find_all("a", href=True):
                    pdf_href = pdf_link["href"]
                    if pdf_href.endswith(".pdf") and f"/{year}" in pdf_href:
                        full_url = base_url + pdf_href if pdf_href.startswith("/") else pdf_href
                        track["paper_links"].append(full_url)
                sibling = sibling.find_next_sibling()

            if track["paper_links"] and "bib" not in title.lower() and not should_skip_title(title):
                grouped_data.append(track)

    # === ACM STYLE ===
    elif "acm" in conf:
        base_url = "https://dl.acm.org"
        session_selectors = [
            "a.section__title",
            "a.accordion-tabbed__control",
        ]
        seen_sessions = set()

        for session_title_tag in soup.select(", ".join(session_selectors)):
            session_title = session_title_tag.get_text(strip=True)
            if not session_title or session_title in seen_sessions or should_skip_title(session_title):
                continue
            seen_sessions.add(session_title)

            parent_div = session_title_tag.find_parent("div", class_="toc__section") or \
                        session_title_tag.find_parent("div", class_="accordion-tabbed__tab") or \
                        session_title_tag.find_parent("section") or \
                        session_title_tag.find_parent("li")

            paper_links = []
            if parent_div:
                input_tag = parent_div.find("input", class_="section--dois", type="hidden")
                if input_tag and input_tag.has_attr("value"):
                    paper_dois = [doi.strip() for doi in input_tag["value"].split(",") if doi.strip()]
                    paper_links.extend(f"{base_url}/doi/{doi}" for doi in paper_dois)

                for doi_link in parent_div.find_all("a", href=True):
                    href = doi_link["href"].strip()
                    if "/doi/" in href:
                        paper_links.append(urljoin(base_url, href))

            paper_links = sorted(set(paper_links))
            if not paper_links:
                continue

            grouped_data.append({
                "track_title": session_title,
                "track_url": None,
                "paper_links": paper_links
            })

        if not grouped_data:
            paper_links = sorted({
                urljoin(base_url, link["href"].strip())
                for link in soup.find_all("a", href=True)
                if "/doi/" in link["href"]
            })
            if paper_links:
                grouped_data.append({
                    "track_title": "All papers",
                    "track_url": None,
                    "paper_links": paper_links
                })

    else:
        raise ValueError(f"[ERROR] Unsupported grouped conference: {conference}")

    # === Save Output JSON ===
    save_dir = Path(f"data/links_raw/{conference}/{year}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "grouped_links.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=2)

    total_links = sum(len(track.get("paper_links", [])) for track in grouped_data)
    print(f"[INFO] ✅ Extracted {total_links} paper links across {len(grouped_data)} tracks.")
    print(f"[INFO] 📁 Saved grouped links to: {save_path.resolve()}")

    return str(save_path.resolve())
