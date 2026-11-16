# Asking SENTRY research questions

The property graph and dense vectors live in `data/rag/sentry.sqlite3`. SQLite
stores nodes, edges and embedding vectors; there is no separate Neo4j, Chroma
or Qdrant service. The graph covers organizations/geography, authors, papers,
conference events and research topics. Semantic search uses
`sentence-transformers/all-MiniLM-L6-v2` and exact cosine similarity.

## Synchronize saved papers

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python -m rag.cli sync --index
```

This scans every `data/final_output/<conference>/<year>/indian_papers_structured.json`,
deduplicates URLs and updates changed graph records and embeddings. Existing
`abstracts_cache.json` entries are matched by exact URL; cache checksums are kept
as provenance. No scraping or generative LLM is involved. Drop `HF_HUB_OFFLINE=1`
on a new machine that needs to download the embedding model.

The full sync on 2026-10-28 found **485 records across 32 folders**: 480 valid
unique papers, four duplicate URL records and one title-less FOCS record excluded
from semantic indexing. There are 87 cached abstracts; the remaining 393 papers
are searched by title. Four duplicate URLs occur in both CCS and SIGCOMM. Their
venue remains unknown, with both source paths retained, rather than choosing a
folder as truth. `2026v1`/`2026v2` remain source labels and normalize to year 2026.

Read `data/rag/corpus_sync_report.json` or `/v1/rag/status` for current coverage.
Source files are untouched. These are earlier SENTRY predicted-Indian papers,
not all conference papers or independently verified ground truth. Deleted files
do not delete historical KG observations. The report describes the latest sync.
After approving institutions, run `sync --index --refresh-resolution` to resolve
unchanged records again. For new extraction runs, enable `SENTRY_RAG_ENABLED=1`
and optionally `SENTRY_RAG_AUTO_INDEX=1`, or run sync afterward.

## Chat through SENTRY

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python orchestrator_cli.py
```

Example questions:

- Who are the authors working on GNN in India? Show supporting papers.
- Find Indian-affiliated graph neural network papers at AAAI in 2026.
- Find papers about language-model hallucination detection.
- Show publications associated with Indian Institute of Science.
- How many papers are indexed, and is the vector index up to date?

The orchestrator uses `query_scholarly_knowledge` and `get_knowledge_status`.
Chat still needs the configured `LLM_*` provider. It should retrieve existing
evidence instead of starting extraction. The tools are also available through
the existing `/v1/chat/completions` endpoint and OpenWebUI connection.
**Restart an already-running SENTRY API process to load the new tools/routes.**

## Direct HTTP tests without a generative LLM

Start the API in another terminal:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python orchestrator_api.py
```

Check coverage:

```bash
curl http://localhost:8091/v1/rag/status \
  -H "Authorization: Bearer $SENTRY_API_KEY"
```

Ask an author question:

```bash
curl http://localhost:8091/v1/rag/query \
  -H "Authorization: Bearer $SENTRY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Who works on GNN in India?","semantic_query":"graph neural networks","target":"authors","mode":"hybrid","k":5}'
```

Set the shell variable `SENTRY_API_KEY` to one of the server's configured
`SENTRY_API_KEYS`. Omit the Authorization header when server auth is disabled.
Both routes use the same authentication as chat.

Other request bodies:

```json
{"question":"Find hallucination detection papers","mode":"vector","target":"papers","semantic_query":"language model hallucination detection","k":5}
```

```json
{"question":"Indian Institute of Science publications","mode":"graph","filters":{"year_min":2025},"k":10}
```

```json
{"question":"Who works on GNN in India?","semantic_query":"graph neural networks","target":"authors","filters":{"country":"India","india_evidence":"trusted"},"k":10}
```

Responses include an extractive answer, supporting papers/citations/URLs, matching
authors and raw affiliations, applied filters, KG version and similarity scores.
`k` limits papers (1–20), not authors. `semantic_query` focuses topic retrieval;
named entities in the full question remain constraints in graph/hybrid modes.
GNN/GNNs expands to graph neural networks. Direct HTTP inference covers common
India phrasing and author intent; supply year/conference/state filters explicitly.
Chat's LLM can construct these filters from a natural-language question.

`min_similarity` defaults to 0.25 for vector/hybrid retrieval. It is an initial
retrieval cutoff, not calibrated confidence or proof of topic expertise. Raise
it for stricter matching or lower it to inspect candidates. Graph mode retrieves
explicit relationships without a semantic score. Top-k results do not establish
exhaustive counts; use `rag.cli aggregate` for exact institutional paper counts.

## What “in India” means

The query service defaults to `india_evidence=pipeline` for India questions.
It returns only authors individually flagged by SENTRY's previous extraction,
or authors whose affiliations match trusted institutional evidence. A foreign
coauthor is not labelled Indian just because the paper has an Indian collaborator.
Each author states which evidence was used.

Use `india_evidence=trusted` to require approved, resolved Indian organizations.
This can omit IIT/IISc/ISI affiliations while the supplemental CSV awaits review.
Pending institutions never auto-promote. State/type/institution restrictions
always require matching trusted organizations.

Results describe historical paper affiliations and related papers, not current
employment, nationality, exhaustive rosters or verified specialisms. Same-name
author occurrences remain distinct unless ORCID supports a shared identity.

## Regression checks

```bash
.venv/bin/python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python benchmark/test_rag.py
```

Tests cover authentication, malformed filters, tool dispatch, source conflicts,
incomplete records, author-level geography, entity constraints and semantic cutoffs.
