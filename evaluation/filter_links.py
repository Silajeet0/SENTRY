"""
Create a filtered links.json for partial evaluation runs.

Indices are 1-based to match pipeline logs and paper_number values.
"""
import argparse
import json
from pathlib import Path


def parse_indices(value: str) -> set[int]:
    if not value:
        return set()

    indices = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return indices


def main():
    parser = argparse.ArgumentParser(description="Filter links.json by 1-based paper indices.")
    parser.add_argument("--links", required=True, help="Input links.json path.")
    parser.add_argument("--output", required=True, help="Output filtered links.json path.")
    parser.add_argument("--start-index", type=int, default=1, help="1-based inclusive start index.")
    parser.add_argument("--end-index", type=int, required=True, help="1-based inclusive end index.")
    parser.add_argument(
        "--exclude-indices",
        default="",
        help="Comma-separated 1-based indices or ranges to exclude, e.g. '61,80-82'.",
    )
    args = parser.parse_args()

    links = json.loads(Path(args.links).read_text(encoding="utf-8"))
    excluded = parse_indices(args.exclude_indices)

    filtered = [
        url for index, url in enumerate(links, start=1)
        if args.start_index <= index <= args.end_index and index not in excluded
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    print(
        f"Saved {len(filtered)} links from indices "
        f"{args.start_index}-{args.end_index}, excluding {sorted(excluded)}, to {output}"
    )


if __name__ == "__main__":
    main()
