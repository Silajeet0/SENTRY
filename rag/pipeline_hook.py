"""Optional ingestion boundary. Retrieval failures do not discard extraction results."""
import logging
import os
from dataclasses import asdict


def enabled():
    return os.getenv("SENTRY_RAG_ENABLED", "0").lower() in {"1", "true", "yes"}


def ingest(info, conference, year):
    if not enabled() or info.error or not info.paper_title:
        return
    from kg import KnowledgeGraph
    from .engine import RAGEngine
    kg = None
    try:
        kg = KnowledgeGraph(os.getenv("SENTRY_KG_PATH", "data/rag/sentry.sqlite3"))
        paper_id = kg.ingest_paper(asdict(info), conference, year)
        if os.getenv("SENTRY_RAG_AUTO_INDEX", "0").lower() in {"1", "true", "yes"}:
            # Reuse the model throughout a conference run.
            engine = RAGEngine(kg, _embedder())
            engine.index()
        logging.getLogger(__name__).info("KG ingested %s", paper_id)
    except Exception:
        logging.getLogger(__name__).exception("RAG ingestion/indexing failed for %s; extraction output retained. Retry with rag.cli ingest/index.", info.paper_url)
    finally:
        if kg is not None:
            kg.close()


def _embedder():
    if not hasattr(_embedder, "instance"):
        from .cli import make_embedder
        _embedder.instance = make_embedder(os.getenv("SENTRY_EMBEDDING_BACKEND", "semantic"), os.getenv("SENTRY_EMBEDDING_MODEL"), os.getenv("SENTRY_EMBEDDING_REVISION"))
    return _embedder.instance
