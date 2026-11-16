"""SQLite property graph. All writes are transactional; observations never imply trust."""
import csv
import hashlib
import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.findall(r"\w+", text))


def stable_id(kind, key):
    return f"{kind}_{hashlib.sha256(key.encode()).hexdigest()[:24]}"


def now():
    return datetime.now(timezone.utc).isoformat()


class KnowledgeGraph:
    def __init__(self, path="data/rag/sentry.sqlite3"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, data TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS nodes_kind ON nodes(kind);
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY, src TEXT REFERENCES nodes(id), relation TEXT,
                dst TEXT REFERENCES nodes(id), evidence_id TEXT, data TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS edges_src ON edges(src);
            CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst);
            CREATE INDEX IF NOT EXISTS edges_evidence ON edges(evidence_id);
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, entity_id TEXT REFERENCES nodes(id),
                text TEXT NOT NULL, metadata TEXT NOT NULL, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS vectors (
                document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                model TEXT, digest TEXT, vector TEXT, PRIMARY KEY(document_id,model));
            CREATE TABLE IF NOT EXISTS audit (
                version INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
                action TEXT, entity_id TEXT, previous TEXT, current TEXT);
        """)

    def close(self):
        self.db.close()

    @property
    def version(self):
        return self.db.execute("SELECT coalesce(max(version),0) FROM audit").fetchone()[0]

    def get(self, entity_id):
        row = self.db.execute("SELECT * FROM nodes WHERE id=?", (entity_id,)).fetchone()
        return {"id": row["id"], "kind": row["kind"], **json.loads(row["data"])} if row else None

    def nodes(self, kind=None):
        rows = self.db.execute("SELECT id FROM nodes WHERE kind=? ORDER BY id", (kind,)) if kind else self.db.execute("SELECT id FROM nodes ORDER BY id")
        return [self.get(row[0]) for row in rows]

    def _node(self, entity_id, kind, data):
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        old = self.db.execute("SELECT data FROM nodes WHERE id=?", (entity_id,)).fetchone()
        if old and old[0] == payload:
            return
        if kind == "ORGANIZATION":
            self._organizations_cache = None
        self.db.execute("INSERT INTO nodes VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data", (entity_id, kind, payload))
        self.db.execute("INSERT INTO audit(timestamp,action,entity_id,previous,current) VALUES(?,?,?,?,?)",
                        (now(), "UPDATE" if old else "INSERT", entity_id, old[0] if old else None, payload))

    def _edge(self, src, relation, dst, evidence_id, **data):
        key = stable_id("EDGE", json.dumps([src, relation, dst, evidence_id]))
        self.db.execute("INSERT OR REPLACE INTO edges VALUES(?,?,?,?,?,?)",
                        (key, src, relation, dst, evidence_id, json.dumps(data, sort_keys=True)))

    def _document(self, entity_id, section, text, metadata):
        if not text.strip():
            return
        doc_id = stable_id("DOC", entity_id + ":" + section)
        digest = hashlib.sha256(text.encode()).hexdigest()
        self.db.execute("INSERT INTO documents VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET text=excluded.text,metadata=excluded.metadata,digest=excluded.digest",
                        (doc_id, entity_id, text, json.dumps({**metadata, "section": section}), digest))
        return doc_id

    def import_organizations(self, path):
        """Read the supplied UGC CSV or the documented supplementary schema."""
        path = Path(path)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        count = 0
        by_name = {normalize(n["canonical_name"]): n for n in self.nodes("ORGANIZATION")}
        with path.open(encoding="utf-8-sig", newline="") as handle, self.db:
            reader = csv.DictReader(handle)
            ugc = "Name of the University" in (reader.fieldnames or [])
            if not ugc and not {"entity_id", "canonical_name", "country", "verification_status"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Unsupported organization CSV schema: {path}")
            for line, row in enumerate(reader, start=2):
                name = (row.get("Name of the University") if ugc else row["canonical_name"]).strip()
                if not name:
                    raise ValueError(f"Missing organization name at {path}:{line}")
                # Resolve canonical duplicates across imports; preserve the first stable ID.
                existing = by_name.get(normalize(name))
                entity_id = existing["id"] if existing else (row.get("entity_id") or stable_id("ORG", "ugc:" + normalize(name)))
                aliases = [a.strip() for a in row.get("aliases", "").split("|") if a.strip()]
                status = "TRUSTED" if ugc else row["verification_status"]
                if status not in {"TRUSTED", "NEEDS_REVIEW", "REJECTED"}:
                    raise ValueError(f"Invalid verification_status at {path}:{line}")
                provenance = {"source_type": "UGC_USER_SUPPLIED" if ugc else row.get("source_type", "OFFICIAL_WEBSITE"),
                              "source_url": row.get("source_url", ""), "file": path.name,
                              "sha256": checksum, "row": line, "retrieved_at": row.get("retrieved_at", ""), "verification_status": status}
                if existing and existing["verification_status"] == "TRUSTED" and status != "TRUSTED":
                    # Keep pending supplemental aliases/locations out of trusted records.
                    self._node(stable_id("REVIEW", row.get("entity_id") or normalize(name)), "REVIEW_RECORD",
                               {"organization_id": entity_id, "proposed_record": row, "source": provenance, "verification_status": status})
                    count += 1
                    continue
                sources = list(existing.get("sources", [])) if existing else []
                if provenance not in sources:
                    sources.append(provenance)
                data = {"canonical_name": name, "normalized_name": normalize(name),
                        "entity_type": row.get("entity_type", "UNIVERSITY"), "country": row.get("country", "India"),
                        "state": row.get("state", ""), "city": row.get("city", ""),
                        "address": row.get("Address", ""), "website": row.get("website", ""),
                        "aliases": sorted(set(aliases + (existing.get("aliases", []) if existing else []))),
                        "verification_status": status,
                        "sources": sources, "notes": row.get("notes", "")}
                if existing:
                    for key in ("city", "state", "website", "address"):
                        data[key] = data[key] or existing.get(key, "")
                self._node(entity_id, "ORGANIZATION", data)
                by_name[normalize(name)] = {"id": entity_id, "kind": "ORGANIZATION", **data}
                self.db.execute("DELETE FROM edges WHERE src=? AND relation='LOCATED_IN'", (entity_id,))
                for kind, value in (("COUNTRY", data["country"]), ("STATE", data["state"]), ("CITY", data["city"])):
                    if value:
                        place_id = stable_id(kind, normalize(data["country"] + ":" + (data["state"] if kind == "CITY" else "") + ":" + value))
                        self._node(place_id, kind, {"name": value})
                        self._edge(entity_id, "LOCATED_IN", place_id, entity_id, status=data["verification_status"])
                text = " | ".join([name, *data["aliases"], data["country"], data["state"], data["city"], data["entity_type"]])
                self._document(entity_id, "organization", text, {"kind": "ORGANIZATION", "status": data["verification_status"]})
                count += 1
        return count

    def resolve(self, affiliation):
        """Only unique canonical/qualified alias matches can resolve automatically.

        Fuzzy candidates are suggestions, never evidence of geographic identity.
        Foreign-campus and bare-acronym strings abstain even when an Indian parent exists.
        """
        text = normalize(re.sub(r"\S+@\S+", "", affiliation))
        if not text or text in {"unknown", "none", "n a", "iit", "nit", "isi", "vit"}:
            return {"status": "UNRESOLVED", "matches": []}
        if re.search(r"\b(formerly|former|previously|alumnus|alumni|not affiliated)\b", text):
            return {"status": "UNRESOLVED", "matches": [], "reason": "Temporal or negated affiliation needs review"}
        foreign = re.search(r"\b(uae|united arab emirates|abu dhabi|dubai|zanzibar|tanzania|italy|usa|united states|singapore|china|canada|australia|germany|united kingdom|uk|england|scotland|france|japan|greece|switzerland|netherlands|malaysia|nepal|qatar)\b", text)
        matches, suggestions = [], []
        if getattr(self, "_organizations_cache", None) is None:
            self._organizations_cache = self.nodes("ORGANIZATION")
        for org in self._organizations_cache:
            if org["verification_status"] != "TRUSTED":
                continue
            names = [org["canonical_name"], *org["aliases"]]
            for name in names:
                term = normalize(name)
                if term in {"iit", "nit", "isi", "vit"} or len(term) < 4:
                    continue
                if f" {term} " in f" {text} ":
                    if not (foreign and org["country"] == "India"):
                        matches.append({"entity_id": org["id"], "method": "CANONICAL" if name == names[0] else "ALIAS", "matched_text": name})
                    break
            else:
                similarity = SequenceMatcher(None, text, normalize(org["canonical_name"])).ratio()
                if similarity >= 0.8:
                    suggestions.append({"entity_id": org["id"], "similarity": similarity, "method": "FUZZY_SUGGESTION"})
        matches = list({m["entity_id"]: m for m in matches}.values())
        return {"status": "RESOLVED" if len(matches) == 1 else "UNRESOLVED", "matches": matches,
                "suggestions": sorted(suggestions, key=lambda x: -x["similarity"])[:5]}

    def ingest_paper(self, paper, conference=None, year=None):
        """Ingest a PaperInfo dictionary; author names are scoped to a paper without ORCID."""
        if paper.get("error"):
            raise ValueError("Cannot ingest failed extraction")
        title = paper.get("paper_title", paper.get("title", "")).strip()
        url = paper.get("paper_url", paper.get("url", "")).strip()
        if not title:
            raise ValueError("Paper must have a title")
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", paper.get("doi", "").strip().lower())
        identity = "doi:" + doi if doi else "url:" + url.rstrip("/") if url else "title:" + normalize(title)
        # If a DOI arrives after the URL-only observation, retain its existing ID.
        existing = next((p for p in self.nodes("PAPER") if (doi and p.get("doi") == doi) or (url and p.get("url", "").rstrip("/") == url.rstrip("/"))), None)
        paper_id = existing["id"] if existing else stable_id("PAPER", identity)
        year = year or paper.get("year")
        year = int(year) if year is not None and str(year).isdigit() else None
        data = {"title": title, "url": url, "doi": doi or (existing or {}).get("doi", ""),
                "abstract": paper.get("abstract", ""), "year": year,
                "conference": conference or paper.get("conference", ""),
                "research_area": paper.get("area_of_research", paper.get("research_area", "")),
                "retrieval_source": paper.get("source") or paper.get("raw_content_source", ""),
                "status": "OBSERVED", "raw_record": paper}
        with self.db:
            self._node(paper_id, "PAPER", data)
            previous_edges = [dict(r) for r in self.db.execute("SELECT * FROM edges WHERE evidence_id=? ORDER BY id", (paper_id,))]
            # Replace this paper's observations, never another paper's evidence.
            self.db.execute("DELETE FROM edges WHERE evidence_id=?", (paper_id,))
            if data["conference"]:
                event_id = stable_id("CONF", normalize(data["conference"]) + ":" + str(year))
                self._node(event_id, "CONFERENCE", {"name": data["conference"], "year": year})
                self._edge(paper_id, "PUBLISHED_AT", event_id, paper_id, status="OBSERVED")
            if data["research_area"]:
                topic_id = stable_id("TOPIC", normalize(data["research_area"]))
                self._node(topic_id, "TOPIC", {"name": data["research_area"]})
                self._edge(paper_id, "HAS_TOPIC", topic_id, paper_id, status="OBSERVED")
            for index, author in enumerate(paper.get("all_authors", paper.get("authors", []))):
                name = author.get("name", "").strip()
                if not name:
                    continue
                orcid = author.get("orcid", "").removeprefix("https://orcid.org/")
                author_id = stable_id("AUTH", "orcid:" + orcid if orcid else f"{paper_id}:{index}:{normalize(name)}")
                self._node(author_id, "AUTHOR", {"name": name, "orcid": orcid, "identity_scope": "ORCID" if orcid else "PAPER"})
                self._edge(author_id, "AUTHORED", paper_id, paper_id, status="OBSERVED")
                affiliations = author.get("affiliations", [author.get("affiliation", "")])
                if isinstance(affiliations, str):
                    affiliations = [affiliations]
                for aff in affiliations:
                    if not aff or normalize(aff) in {"unknown", "n a", "none"}:
                        continue
                    resolution = self.resolve(aff)
                    if resolution["status"] == "RESOLVED":
                        target = resolution["matches"][0]["entity_id"]
                        method = resolution["matches"][0]["method"]
                    else:
                        target = stable_id("CAND", normalize(aff))
                        self._node(target, "CANDIDATE", {"canonical_name": aff, "verification_status": "NEEDS_REVIEW"})
                        method = "UNRESOLVED"
                    self._edge(author_id, "AFFILIATED_WITH", target, paper_id,
                               observed_affiliation=aff, observed_year=year, status="OBSERVED", resolution_method=method)
            metadata = {"kind": "PAPER", "year": year, "conference": data["conference"], "status": "OBSERVED"}
            documents = [self._document(paper_id, "summary", title + "\n" + data["abstract"], metadata)]
            words = paper.get("full_text", "").split()
            for start in range(0, len(words), 200):
                documents.append(self._document(paper_id, f"chunk:{start // 200}", " ".join(words[start:start + 240]), metadata))
            for row in self.db.execute("SELECT id FROM documents WHERE entity_id=?", (paper_id,)).fetchall():
                if row[0] not in documents:
                    self.db.execute("DELETE FROM documents WHERE id=?", (row[0],))
            current_edges = [dict(r) for r in self.db.execute("SELECT * FROM edges WHERE evidence_id=? ORDER BY id", (paper_id,))]
            if current_edges != previous_edges:
                self.db.execute("INSERT INTO audit(timestamp,action,entity_id,previous,current) VALUES(?,?,?,?,?)",
                                (now(), "REPLACE_OBSERVATIONS", paper_id, json.dumps(previous_edges), json.dumps(current_edges)))
        return paper_id

    def edges(self, entity_id=None):
        rows = self.db.execute("SELECT * FROM edges WHERE src=? OR dst=? ORDER BY id", (entity_id, entity_id)) if entity_id else self.db.execute("SELECT * FROM edges ORDER BY id")
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]

    def subgraph(self, seeds, depth=2, limit=100):
        if not 0 <= depth <= 4 or not 1 <= limit <= 500:
            raise ValueError("Traversal requires depth 0..4 and limit 1..500")
        seen = {s for s in seeds if self.get(s)}
        if len(seen) > limit:
            raise ValueError("Too many seeds for traversal limit")
        frontier, edges = set(seen), {}
        for _ in range(depth):
            following = set()
            for entity in sorted(frontier):
                for edge in self.edges(entity):
                    target = edge["dst"] if edge["src"] == entity else edge["src"]
                    if target not in seen and len(seen) >= limit:
                        continue
                    edges[edge["id"]] = edge
                    if target not in seen:
                        seen.add(target)
                        following.add(target)
            frontier = following
        return {"nodes": [self.get(i) for i in sorted(seen)], "edges": list(edges.values())}

    def paper_evidence(self, paper_id):
        paper = self.get(paper_id)
        if not paper or paper["kind"] != "PAPER":
            raise ValueError("Unknown paper")
        edges = [e for e in self.edges() if e["evidence_id"] == paper_id]
        ids = {paper_id} | {e[k] for e in edges for k in ("src", "dst")}
        return {"paper": paper, "nodes": [self.get(i) for i in sorted(ids)], "edges": edges}

    def shortest_path(self, source, target, max_depth=4, max_nodes=500):
        """Bounded undirected evidence path; does not infer permanent affiliations."""
        if not 0 <= max_depth <= 6 or not 1 <= max_nodes <= 1000:
            raise ValueError("Path search requires depth 0..6 and node limit 1..1000")
        if not self.get(source) or not self.get(target):
            raise ValueError("Unknown path endpoint")
        frontier = [(source, [])]
        seen = {source}
        for _ in range(max_depth+1):
            following = []
            for current, path in frontier:
                if current == target:
                    return {"found": True, "edges": path, "source": source, "target": target}
                if len(path) == max_depth:
                    continue
                for edge in self.edges(current):
                    node = edge["dst"] if edge["src"] == current else edge["src"]
                    if node not in seen and len(seen) < max_nodes:
                        seen.add(node)
                        following.append((node, path+[edge]))
            frontier = following
        return {"found": False, "edges": [], "source": source, "target": target,
                "note": "No path found within the configured bounds."}
