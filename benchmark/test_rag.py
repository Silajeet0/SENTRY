"""Semantic RAG integration check on an explicitly synthetic corpus; no LLM API call."""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg import KnowledgeGraph
from rag import RAGEngine
from rag.cli import make_embedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-backend", choices=["semantic", "hash"], default="semantic")
    parser.add_argument("--output", type=Path, default=ROOT/"benchmark/results/rag_smoke_report.json")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        csv_path = folder/"organizations.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["entity_id", "canonical_name", "aliases", "country", "state", "verification_status"])
            writer.writerow(["ORG_A", "Fixture University A", "Fixture A", "India", "West Bengal", "TRUSTED"])
            writer.writerow(["ORG_B", "Fixture University B", "Fixture B", "India", "Maharashtra", "TRUSTED"])
        graph = KnowledgeGraph(folder/"test.sqlite3")
        try:
            graph.import_organizations(csv_path)
            fixtures = [
                ("video", "Temporal language grounding", "Locate the start and end of an event described by a sentence in an untrimmed video.", "Vision", ["Fixture A", "Fixture B"], 2026),
                ("database", "Efficient database query execution", "Optimizing relational SQL joins using query plans and table indexes.", "Database", ["Fixture A"], 2025),
                ("foreign", "Video moment retrieval with language", "Search films for a specific scene using natural language.", "Vision", ["Unknown Overseas Lab"], 2026),
            ]
            ids = {}
            for name,title,abstract,topic,orgs,year in fixtures:
                ids[name] = graph.ingest_paper({"paper_url":f"https://example.org/fixture/{name}","paper_title":title,"abstract":abstract,
                    "area_of_research":topic,"year":year,"conference":"SYNTHETIC",
                    "all_authors":[{"name":f"Synthetic Author {name} {i}","affiliation":org} for i,org in enumerate(orgs)]})
            embedder = make_embedder(args.embedding_backend)
            engine = RAGEngine(graph, embedder)
            indexed = engine.index()
            vector = engine.search("Find the scene in a movie that matches a written description", mode="vector", filters={"country":"India"})
            hybrid = engine.search("video moment retrieval", mode="hybrid", filters={"country":"India", "year_min":2026})
            related = engine.search("Fixture A and Fixture B collaboration", mode="graph")
            answer = engine.answer("Fixture A video", mode="hybrid", filters={"year":2026})
            checks = {"semantic_paraphrase":vector[0]["entity_id"] == ids["video"],
                      "hybrid_excludes_unverified_organizations":[r["entity_id"] for r in hybrid] == [ids["video"]],
                      "graph_collaboration":[r["entity_id"] for r in related] == [ids["video"]],
                      "citations":bool(answer["citations"]), "incremental_index_noop":engine.index() == 0}
            result = {"corpus":"Synthetic fixture; not a retrieval quality benchmark", "embedding":embedder.identity,
                      "indexed_documents":indexed,"checks":checks,"answer":answer,
                      "vector_ranking":[{"paper_id":h["entity_id"],"score":h["score"]} for h in vector]}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2)+"\n")
            print(json.dumps(checks,indent=2))
            if not all(checks.values()):
                raise SystemExit(1)
        finally:
            graph.close()


if __name__ == "__main__":
    main()
