import json
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from workflows.link_extractors.skip_patterns import should_skip_title


def _extract_generic_heading_grouped(soup: BeautifulSoup, proceeding_url: str = "") -> list:
    """
    Best-effort fallback for single-page, track-grouped proceedings that don't
    match a dedicated extractor (ACL/ACM/AAAI). Walks the page in document
    order, treating any heading tag (h1-h5) as a new track boundary, and
    collects every '.pdf'-suffixed link seen before the next heading as
    belonging to that track. Headings with no PDF links under them (nav/site
    title headings, empty umbrella headings like "Main Track") are dropped
    automatically since they end up with an empty paper_links list.
    """
    grouped_data = []
    current_track = None

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "a"]):
        if element.name in ("h1", "h2", "h3", "h4", "h5"):
            title = element.get_text(strip=True)
            if not title or should_skip_title(title):
                current_track = None
                continue
            current_track = {"track_title": title, "track_url": None, "paper_links": []}
            grouped_data.append(current_track)
        elif current_track is not None:
            href = (element.get("href") or "").strip()
            if href.lower().endswith(".pdf"):
                current_track["paper_links"].append(urljoin(proceeding_url, href))

    grouped_data = [t for t in grouped_data if t["paper_links"]]

    if not grouped_data:
        paper_links = sorted({
            urljoin(proceeding_url, a["href"].strip())
            for a in soup.find_all("a", href=True)
            if a["href"].strip().lower().endswith(".pdf")
        })
        if paper_links:
            grouped_data.append({
                "track_title": "All papers",
                "track_url": None,
                "paper_links": paper_links
            })

    return grouped_data


def extract_grouped_links(html_path: str, conference: str, year: str, proceeding_url: str="") -> str:
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
    domain = urlparse(proceeding_url or "").netloc.lower()

    # ACL STYLE
    if domain == "aclanthology.org" or "acl" in conf:
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

    # ACM STYLE
    elif "acm.org" in domain or "acm" in conf:
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

    elif "aaai" in conf:
        raise ValueError(
            "[ERROR] AAAI proceedings use a two-level landing-page → "
            "per-volume-OJS-issue structure that this generic single-page "
            "extractor can't handle. This should already be routed around "
            "in main_driver.run_pipeline — use "
            "workflows.link_extractors.aaai_link_extractor.extract_aaai_links "
            "directly instead."
        )

    else:
        print(
            f"[WARN] No dedicated extractor for '{conference}' — "
            "falling back to generic single-page heading-grouped extraction. "
            "Spot-check the resulting track/paper counts before trusting this run."
        )
        grouped_data = _extract_generic_heading_grouped(soup, proceeding_url)
        if not grouped_data:
            raise ValueError(
                f"[ERROR] Unsupported grouped conference: {conference}. "
                "Generic fallback found no heading + PDF-link structure either — "
                "this venue needs a dedicated extractor."
            )

    # Save Output JSON
    save_dir = Path(f"data/links_raw/{conference}/{year}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "grouped_links.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(grouped_data, f, indent=2)

    total_links = sum(len(track.get("paper_links", [])) for track in grouped_data)
    print(f"[INFO] Extracted {total_links} paper links across {len(grouped_data)} tracks.")
    print(f"[INFO] Saved grouped links to: {save_path.resolve()}")

    return str(save_path.resolve())