"""Shared RAG service for SENTRY chat tools and authenticated HTTP endpoints."""
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from threading import RLock

from kg import KnowledgeGraph
from kg.store import normalize
from .cli import make_embedder
from .engine import RAGEngine

ROOT = Path(__file__).resolve().parents[1]
_MODEL_LOCK = RLock()


def database_path():
    return Path(os.getenv("SENTRY_KG_PATH", str(ROOT / "data/rag/sentry.sqlite3")))


def _open_graph():
    path = database_path()
    if not path.is_file():
        raise FileNotFoundError("RAG database is missing. Run rag.cli bootstrap, then rag.cli sync --index.")
    return KnowledgeGraph(path)


@lru_cache(maxsize=2)
def _load_embedder(backend, model, revision):
    return make_embedder(backend, model, revision)


def get_embedder():
    with _MODEL_LOCK:
        return _load_embedder(os.getenv("SENTRY_EMBEDDING_BACKEND", "semantic"),
                              os.getenv("SENTRY_EMBEDDING_MODEL"), os.getenv("SENTRY_EMBEDDING_REVISION"))


def knowledge_status():
    if not database_path().is_file():
        return {"status": "not_initialized", "database": str(database_path()),
                "next_step": "Run rag.cli bootstrap, then rag.cli sync --index."}
    graph = _open_graph()
    try:
        counts = dict(graph.db.execute("SELECT kind,count(*) FROM nodes GROUP BY kind").fetchall())
        documents = graph.db.execute("SELECT count(*) FROM documents").fetchone()[0]
        vectors = []
        for row in graph.db.execute("SELECT model,count(*) FROM vectors GROUP BY model"):
            fresh = graph.db.execute("SELECT count(*) FROM vectors v JOIN documents d ON d.id=v.document_id WHERE v.model=? AND v.digest=d.digest", (row[0],)).fetchone()[0]
            vectors.append({"model": row[0], "vectors": row[1], "fresh_vectors": fresh, "missing_or_stale": documents-fresh})
        reports = []
        if graph.db.execute("SELECT 1 FROM sqlite_master WHERE name='corpus_sync'").fetchone():
            for row in graph.db.execute("SELECT report FROM corpus_sync"):
                report = json.loads(row[0])
                reports.append({k:v for k,v in report.items() if k not in {"files", "paper_ids"}})
        return {"status": "ready" if counts.get("PAPER") and any(v["missing_or_stale"] == 0 for v in vectors) else "needs_indexing",
                "storage": "SQLite property graph and dense-vector tables; exact cosine search",
                "database": str(database_path()), "kg_version": graph.version, "nodes": counts,
                "documents": documents, "vector_indexes": vectors, "corpus_imports": reports,
                "coverage_note": "Import reports describe the last sync. Run sync again after new extraction outputs."}
    finally:
        graph.close()


def _author_rows(evidence, filters, citation):
    nodes = {n["id"]: n for n in evidence["nodes"]}
    listed = {normalize(n) for n in evidence["paper"].get("raw_record", {}).get("authors_with_indian_affiliations", [])}
    rows = []
    for author in nodes.values():
        if author["kind"] != "AUTHOR":
            continue
        if filters.get("author_id") and author["id"] != filters["author_id"]:
            continue
        affiliations, trusted_match = [], False
        org_filters = {k:filters[k] for k in ("country", "state", "organization_id", "organization_type") if k in filters}
        for edge in evidence["edges"]:
            if edge["src"] != author["id"] or edge["relation"] != "AFFILIATED_WITH":
                continue
            org = nodes[edge["dst"]]
            trusted = org["kind"] == "ORGANIZATION" and org["verification_status"] == "TRUSTED"
            mapping = {"organization_id":"id", "organization_type":"entity_type"}
            matched = trusted and all(normalize(org.get(mapping.get(k,k), "")) == normalize(v) for k,v in org_filters.items())
            trusted_match |= matched
            affiliations.append({"observed": edge["data"]["observed_affiliation"], "organization_id": org["id"],
                                 "trusted_organization": trusted, "matches_filters": matched})
        pipeline_match = (org_filters == {"country":"India"} and filters.get("india_evidence") == "pipeline" and normalize(author["name"]) in listed)
        if org_filters and not (trusted_match or pipeline_match):
            continue
        rows.append({"author_id":author["id"], "name":author["name"], "identity_scope":author["identity_scope"],
                     "affiliations":affiliations, "geographic_evidence":"trusted_organization" if trusted_match else "sentry_pipeline_observation" if pipeline_match else "not_filtered",
                     "citation":citation, "paper_id":evidence["paper"]["id"]})
    return rows


def query_knowledge(question, mode="hybrid", target="auto", filters=None, k=5, semantic_query=None, min_similarity=0.25):
    """Return bounded, cited evidence; chat's existing LLM can synthesize this result.

    This endpoint needs no generative LLM. Semantic similarity is a retrieval score,
    not proof that a person currently works on a topic.
    """
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("question must contain 1..2000 characters")
    if mode not in {"vector", "graph", "hybrid"} or target not in {"auto", "authors", "papers", "organizations"}:
        raise ValueError("Invalid mode or target")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20")
    if not isinstance(min_similarity, (int,float)) or not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be between 0 and 1")
    if semantic_query is not None and (not isinstance(semantic_query,str) or not semantic_query.strip() or len(semantic_query)>2000):
        raise ValueError("semantic_query must contain 1..2000 characters")
    if filters is not None and not isinstance(filters, dict):
        raise ValueError("filters must be an object")
    filters = dict(filters or {})
    # Keep the geographic constraint deterministic even when the chat model
    # omits an explicit filters object. Explicit filters still take priority.
    if re.search(
        r"\b(?:in|from|within|based in|located in)\s+india\b"
        r"|\bindian(?:[-\s]affiliated)?\b",
        question,
        re.I,
    ):
        filters.setdefault("country", "India")
    if normalize(filters.get("country", "")) == "india":
        filters["country"] = "India"
        filters.setdefault("india_evidence", "pipeline")
    if target == "auto":
        target = "authors" if re.search(r"\b(who|authors?|researchers?)\b", question, re.I) else "papers"
    # An author-only lookup is an exact graph query.  Running it through the
    # hybrid semantic threshold can discard valid papers simply because the
    # query text (for example, "recent papers") is not close to the paper's
    # abstract.  A topic can still be supplied explicitly through
    # ``semantic_query``; in that case hybrid retrieval remains intentional.
    author_only_lookup = bool(filters.get("author_name")) and not semantic_query
    if author_only_lookup and mode == "hybrid":
        mode = "graph"
    query = semantic_query or question
    query = re.sub(r"\bGNNs?\b", "graph neural networks", query, flags=re.I)
    query = re.sub(r"\bRL\b", "reinforcement learning", query, flags=re.I)
    query = re.sub(r"\bNNs?\b", "neural networks", query, flags=re.I)
    graph = _open_graph()
    try:
        engine = RAGEngine(graph, get_embedder() if mode != "graph" else None)
        with _MODEL_LOCK:
            hits = engine.search(query, mode=mode, filters=filters, k=k, min_similarity=min_similarity, entity_query=question)
        papers, authors, citations, organizations = [], [], {}, {}
        for hit in hits:
            evidence = hit["evidence"]
            paper = evidence["paper"]
            citation = f"P{len(papers)+1}"
            citations[citation] = {"paper_id":paper["id"], "url":paper["url"], "title":paper["title"]}
            papers.append({"paper_id":paper["id"], "title":paper["title"], "abstract":paper["abstract"][:800],
                           "year":paper["year"], "conference":paper["conference"], "citation":citation,
                           "similarity":hit.get("vector_score", hit["score"] if mode == "vector" else None)})
            matched_authors = _author_rows(evidence, filters, citation)
            authors.extend(matched_authors)
            for author in matched_authors:
                for aff in author["affiliations"]:
                    if aff["trusted_organization"] and (not filters.get("country") or aff["matches_filters"]):
                        org = graph.get(aff["organization_id"])
                        organizations.setdefault(org["id"], {"organization_id":org["id"], "name":org["canonical_name"], "citations":[]})["citations"].append(citation)
        # Keep a chat tool result bounded, without pretending the list is exhaustive.
        returned_authors = authors[:100]
        if target == "authors":
            lines = [f"{a['name']} — {'; '.join(x['observed'] for x in a['affiliations']) or 'affiliation not recorded'} [{a['citation']}]" for a in returned_authors]
        elif target == "organizations":
            lines = [f"{o['name']} {' '.join('['+c+']' for c in sorted(set(o['citations'])))}" for o in organizations.values()]
        else:
            lines = [f"{p['title']} [{p['citation']}]" for p in papers]
        return {"status":"ok", "question":question, "semantic_query":query, "mode":mode, "target":target,
                "filters":filters, "kg_version":graph.version, "min_similarity":min_similarity,
                "answer":"\n".join(lines) if lines else "No matching evidence found in the indexed corpus under these filters.",
                "papers":papers, "authors":returned_authors, "authors_truncated":len(authors)>len(returned_authors),
                "organizations":list(organizations.values()), "citations":citations,
                "limitations":["Top-k related papers, not an exhaustive roster or proof of a current job or nationality.",
                    "Pipeline India labels are observations and can be wrong; use india_evidence=trusted for KG-confirmed institutions.",
                    "Same-name author occurrences are not merged without ORCID. Papers without abstracts are searched by title."]}
    finally:
        graph.close()
