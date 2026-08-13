import json
from pathlib import Path

def select_tracks_cli(grouped_json_path: str, output_dir: str = None) -> str:
    """
    CLI-based track selector for grouped paper links.

    Args:
        grouped_json_path (str): Path to JSON file with grouped tracks
        output_dir (str): Optional directory to save selected links

    Returns:
        str: Path to the saved flat JSON with selected track paper links
    """
    # Load grouped JSON
    with open(grouped_json_path, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    print(f"\n Loaded {len(tracks)} tracks from: {grouped_json_path}\n")

    # Display track titles with indices
    for idx, track in enumerate(tracks):
        print(f"[{idx}] {track['track_title']}  ({len(track['paper_links'])} papers)")

    # Get user selection
    selected_indices = input("\nEnter the indices of the tracks you want to select (comma-separated): ")
    selected_indices = [int(i.strip()) for i in selected_indices.split(",") if i.strip().isdigit()]

    # Filter selected tracks
    selected_tracks = [tracks[i] for i in selected_indices if 0 <= i < len(tracks)]

    # Identify conference
    grouped_path = Path(grouped_json_path)
    conference = grouped_path.parts[-3].lower()
    year = grouped_path.parts[-2]

    # Remove paper 0 PDFs for ACL (links ending with '.0.pdf')
    cleaned_links = []
    for track in selected_tracks:
        for link in track["paper_links"]:
            if not (conference == "acl" and link.endswith(".0.pdf")):
                cleaned_links.append(link)

    # Deduplicate and sort
    all_paper_links = sorted(set(cleaned_links))

    # Save output path
    base_dir = output_dir or grouped_path.parent.parent / "selected_links"
    save_dir = Path(base_dir) / conference / year
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "links.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_paper_links, f, indent=2)

    print(f"\n Saved {len(all_paper_links)} links from {len(selected_tracks)} track(s) → {save_path.resolve()}\n")
    return str(save_path.resolve())
