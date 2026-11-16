"""Run with python -m rag.cli. No service or model is loaded for graph-only commands."""
import argparse
import json
import os
from pathlib import Path

from kg import KnowledgeGraph
from .engine import RAGEngine


def make_embedder(backend="semantic", model=None, revision=None):
    from .embeddings import HashEmbedder, SentenceTransformerEmbedder
    if backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(model or "sentence-transformers/all-MiniLM-L6-v2", revision)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/rag/sentry.sqlite3")
    parser.add_argument("--embedding-backend", choices=["semantic", "hash"], default="semantic")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-revision")
    sub = parser.add_subparsers(dest="command", required=True)
    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("csv", nargs="+", type=Path)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("json", type=Path)
    ingest.add_argument("--conference")
    ingest.add_argument("--year", type=int)
    sub.add_parser("index")
    sync = sub.add_parser("sync")
    sync.add_argument("--root", default="data/final_output")
    sync.add_argument("--report", type=Path, default=Path("data/rag/corpus_sync_report.json"))
    sync.add_argument("--index", action="store_true", help="Update embeddings after backfilling all saved outputs")
    sync.add_argument("--refresh-resolution", action="store_true", help="Resolve unchanged records again after reviewing institution data")
    resolve = sub.add_parser("resolve")
    resolve.add_argument("affiliation")
    graph = sub.add_parser("subgraph")
    graph.add_argument("entity_ids", nargs="+")
    graph.add_argument("--depth", type=int, default=2)
    graph.add_argument("--limit", type=int, default=100)
    path = sub.add_parser("path")
    path.add_argument("source")
    path.add_argument("target")
    path.add_argument("--depth", type=int, default=4)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--filters", default="{}")
    query = sub.add_parser("query")
    query.add_argument("question")
    query.add_argument("--mode", choices=["vector", "graph", "hybrid"], default="hybrid")
    query.add_argument("--filters", default="{}", help="JSON object of explicit graph constraints")
    query.add_argument("--entity", action="append")
    query.add_argument("-k", type=int, default=10)
    query.add_argument("--generate", action="store_true", help="Synthesize using configured LLM; otherwise return cited evidence")
    args = parser.parse_args(argv)
    kg = KnowledgeGraph(args.db)
    try:
        embedder = make_embedder(args.embedding_backend, args.embedding_model, args.embedding_revision) if args.command == "index" or (args.command == "sync" and args.index) or (args.command == "query" and args.mode != "graph") else None
        engine = RAGEngine(kg, embedder)
        if args.command == "sync":
            from .corpus import sync_outputs
            result = sync_outputs(kg, args.root, args.refresh_resolution)
            if args.index:
                result["indexed_documents"] = engine.index()
                result["embedding_model"] = embedder.identity
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2)+"\n")
            result = {k:v for k,v in result.items() if k not in {"paper_ids", "files"}}
        elif args.command == "bootstrap":
            result = {str(p): kg.import_organizations(p) for p in args.csv}
        elif args.command == "ingest":
            papers = json.loads(args.json.read_text(encoding="utf-8"))
            if not isinstance(papers, list):
                raise ValueError("Input must be a JSON array of paper records")
            result = {"paper_ids": [kg.ingest_paper(p, args.conference, args.year) for p in papers]}
        elif args.command == "index":
            result = {"indexed_documents": engine.index(), "embedding_model": embedder.identity}
        elif args.command == "resolve":
            result = kg.resolve(args.affiliation)
        elif args.command == "subgraph":
            result = kg.subgraph(args.entity_ids, args.depth, args.limit)
        elif args.command == "path":
            result = kg.shortest_path(args.source, args.target, args.depth)
        elif args.command == "aggregate":
            result = engine.aggregate(json.loads(args.filters))
        else:
            generator = None
            if args.generate:
                from dotenv import load_dotenv
                from openai import OpenAI
                load_dotenv()
                client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"), timeout=60)
                def generator(prompt):
                    return client.chat.completions.create(model=os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct"),
                        messages=[{"role": "system", "content": "Use only supplied evidence and cite each factual claim."}, {"role": "user", "content": prompt}],
                        temperature=0, max_tokens=1500).choices[0].message.content or ""
            result = engine.answer(args.question, generator=generator, mode=args.mode, k=args.k,
                                   filters=json.loads(args.filters), entities=args.entity)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        kg.close()


if __name__ == "__main__":
    main()
