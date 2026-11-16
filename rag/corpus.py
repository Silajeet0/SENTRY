"""Idempotent backfill of saved SENTRY outputs, with an auditable coverage report."""
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from kg.store import now


def sync_outputs(graph, root="data/final_output", refresh_resolution=False):
    root = Path(root)
    files = sorted(root.glob("*/*/indian_papers_structured.json"))
    if not files:
        raise ValueError(f"No saved extraction outputs under {root}")
    grouped, errors, source_files = defaultdict(list), [], []
    abstracts, cache_sources = {}, []
    for path in sorted(root.glob("*/*/abstracts_cache.json")):
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                raise ValueError("Expected an abstract cache object")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            cache_sources.append({"file":str(path), "sha256":digest})
            for url, entry in cache.items():
                if isinstance(entry,dict) and isinstance(entry.get("abstract"),str) and entry["abstract"].strip() and not entry.get("error"):
                    if entry.get("paper_url",url).rstrip("/") != url.rstrip("/"):
                        continue
                    abstracts[url.rstrip("/")] = (entry["abstract"], {"file":str(path), "sha256":digest, "key":url})
        except (ValueError, UnicodeError) as exc:
            errors.append({"file":str(path), "error":str(exc)})
    existing_by_url = {p["url"].rstrip("/"):p for p in graph.nodes("PAPER") if p.get("url")}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("Expected a JSON array")
        except (ValueError, UnicodeError) as exc:
            errors.append({"file": str(path), "error": str(exc)})
            continue
        source_files.append({"file": str(path), "sha256": digest, "records": len(rows)})
        for index, row in enumerate(rows):
            source = {"file": str(path), "row": index, "sha256": digest,
                      "conference": path.parent.parent.name, "year_label": path.parent.name}
            if not isinstance(row, dict):
                errors.append({**source, "error": "Paper record is not an object"})
                continue
            key = (row.get("paper_url") or "").rstrip("/")
            if not key:
                errors.append({**source, "error": "Missing paper URL; cannot track corpus identity"})
                continue
            grouped[key].append((row, source))
    imported, conflicts = {}, []
    for url, observations in grouped.items():
        # Prefer a complete record, while preserving every source observation.
        record = dict(max((r for r, _ in observations), key=lambda r: (bool(r.get("paper_title")), bool(r.get("abstract")), len(r.get("all_authors", [])))))
        sources = [s for _, s in observations]
        conferences = {s["conference"] for s in sources}
        years = {int(m.group()) for s in sources if (m := re.match(r"\d{4}", s["year_label"]))}
        conference = next(iter(conferences)) if len(conferences) == 1 else ""
        if len(conferences) > 1:
            conflicts.append({"paper_url": url, "conferences": sorted(conferences),
                              "policy": "Venue left unknown; source folders conflict."})
        record["corpus_sources"] = sources
        if not record.get("abstract") and url in abstracts:
            record["abstract"], record["abstract_cache_source"] = abstracts[url]
        record["conference"] = conference
        record["year"] = next(iter(years)) if len(years) == 1 else None
        # Preserve upstream India decisions as observations, never as trusted KG facts.
        record["authors_with_indian_affiliations"] = sorted({name for r, _ in observations for name in r.get("authors_with_indian_affiliations", [])})
        try:
            existing = existing_by_url.get(url)
            if not refresh_resolution and existing and existing["raw_record"] == record:
                paper_id = existing["id"]
            else:
                paper_id = graph.ingest_paper(record, conference=conference, year=record["year"])
            imported[url] = paper_id
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append({"paper_url": url, "sources": sources, "error": str(exc)})
    report = {"updated_at": now(), "root": str(root.resolve()), "files": source_files,
              "source_files": len(files), "source_records": sum(f["records"] for f in source_files),
              "abstract_cache_sources": cache_sources,
              "unique_source_urls": len(grouped), "ingested_unique_papers": len(set(imported.values())),
              "duplicate_records": sum(len(v)-1 for v in grouped.values()),
              "missing_abstracts": sum(not graph.get(i)["abstract"] for i in set(imported.values())),
              "errors": errors, "venue_conflicts": conflicts, "paper_ids": sorted(set(imported.values())),
              "scope": "Saved SENTRY predicted-Indian outputs; not independently verified ground truth."}
    graph.db.execute("CREATE TABLE IF NOT EXISTS corpus_sync (root TEXT PRIMARY KEY, report TEXT NOT NULL)")
    with graph.db:
        graph.db.execute("INSERT OR REPLACE INTO corpus_sync VALUES(?,?)", (str(root.resolve()), json.dumps(report)))
    return report
