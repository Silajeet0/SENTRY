import json
from pathlib import Path
from typing import Optional


def select_tracks_auto(
    grouped_json_path: str,
    skip_keywords: Optional[list[str]] = None,
    include_keywords: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Non-interactive counterpart to utils.track_selector_cli.select_tracks_cli.

    track_selector_cli blocks on input() to ask a human which tracks to
    keep — fine for a terminal session, fatal for an agent driving the
    pipeline unattended (the process just hangs forever waiting on stdin
    that's never coming). This does the same track filtering + saving, but
    decides which tracks to keep by keyword match instead of a prompt.

    Args:
        grouped_json_path: path to a grouped_links.json (list of
            {"track_title", "track_url", "paper_links"} dicts).
        skip_keywords: track titles containing any of these (case-insensitive
            substring match) are excluded, e.g. ["workshop", "tutorial"].
        include_keywords: if given, ONLY tracks containing at least one of
            these are kept (applied before skip_keywords). Leave empty/None
            to start from "everything".
        output_dir: same override as select_tracks_cli; defaults to
            <grouped_json's grandparent>/selected_links so output lands in
            the exact same place the CLI version would.

    Returns:
        dict with the saved path, and which tracks were kept/skipped, so
        callers (and the LLM orchestrator) can report exactly what happened
        instead of it being invisible.
    """
    with open(grouped_json_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    skip_kw = [k.lower() for k in (skip_keywords or [])]
    include_kw = [k.lower() for k in (include_keywords or [])]

    def _wanted(title: str) -> bool:
        t = (title or "").lower()
        if include_kw and not any(k in t for k in include_kw):
            return False
        if any(k in t for k in skip_kw):
            return False
        return True

    selected_tracks = []
    skipped_titles = []
    for track in tracks:
        title = track.get("track_title", "")
        if _wanted(title):
            selected_tracks.append(track)
        else:
            skipped_titles.append(title)

    grouped_path = Path(grouped_json_path)
    conference = grouped_path.parts[-3].lower()
    year = grouped_path.parts[-2]

    # Same ACL paper-0 PDF exclusion as the interactive version.
    cleaned_links = []
    for track in selected_tracks:
        for link in track.get("paper_links", []):
            if not (conference == "acl" and link.endswith(".0.pdf")):
                cleaned_links.append(link)

    all_paper_links = sorted(set(cleaned_links))

    base_dir = output_dir or grouped_path.parent.parent / "selected_links"
    save_dir = Path(base_dir) / conference / year
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "links.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_paper_links, f, indent=2)

    return {
        "path": str(save_path.resolve()),
        "total_tracks": len(tracks),
        "selected_tracks": [t.get("track_title", "") for t in selected_tracks],
        "skipped_tracks": skipped_titles,
        "total_links": len(all_paper_links),
    }
