# SENTRY RAG extension

For conversational queries, HTTP endpoints and complete-corpus backfilling,
see [QUERYING.md](QUERYING.md). RAG is now connected to SENTRY's orchestrator.

This implements the retrieval scope of the supplied architecture: an institution
and scholarly property graph, semantic vectors, GraphRAG, HybridRAG, provenance
and grounded answers. Production affiliation decisions use extracted affiliation
strings and the fine-tuned `laya_finetuning/laya_finetuned_SENTRY` checkpoint.
The deterministic classifier is not part of the production path.
Candidate review UI, entity merging, Laya decision fusion, automatic discovery,
and the other later architecture extensions are not implemented here.

## Install and bootstrap

```bash
# CPU machine; choose an appropriate PyTorch build if using a GPU.
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt -r requirements-rag.txt
.venv/bin/python -m rag.cli bootstrap KG_Data/Universities.csv KG_Data/supplementary_institutions.csv
```

The SQLite database defaults to `data/rag/sentry.sqlite3`; override with the global
`--db` option. New supplement records remain pending until you review them; see
`KG_Data/README.md`. No database server or hosted vector service is needed.

Import existing pipeline output, then build semantic vectors:

```bash
.venv/bin/python -m rag.cli ingest data/final_output/ACL/2025/indian_papers_structured.json --conference ACL --year 2025
.venv/bin/python -m rag.cli index
.venv/bin/python -m rag.cli query 'language model hallucination detection' --mode vector
.venv/bin/python -m rag.cli query 'Jadavpur University collaborations' --mode graph
.venv/bin/python -m rag.cli query 'multimodal retrieval' --mode hybrid --filters '{"country":"India","year_min":2025}'
.venv/bin/python -m rag.cli aggregate --filters '{"conference":"ACL","year":2025}'
```

`index` lazily downloads `sentence-transformers/all-MiniLM-L6-v2` into
`data/model_cache`. `--embedding-model` and `--embedding-revision` select another
checkpoint. Model identity includes the resolved revision; changing it requires
indexing for that model. `HF_HUB_OFFLINE=1` runs with cached weights. The explicit
global `--embedding-backend hash` provides lexical vectors for offline tests;
it is **not semantic retrieval** and never silently replaces the real model.

Queries return citations and structured evidence by default. Add `--generate`
to synthesize an answer with the existing `LLM_API_KEY`, `LLM_BASE_URL` and
`LLM_MODEL` configuration. This makes an external LLM request. Evidence is bounded
and marked as data; the prompt requires citations and uncertainty. Unknown or
missing citation IDs are rejected, but citation ID validation cannot establish
that each generated claim is entailed by its evidence. No external generation
was needed for the included verification run.

## Pipeline ingestion

```bash
export SENTRY_RAG_ENABLED=1
export SENTRY_KG_PATH=data/rag/sentry.sqlite3
# Optional: update embeddings immediately after each successful paper.
export SENTRY_RAG_AUTO_INDEX=1
```

Then use the existing conference entry points. Successful extractions enter the
graph before the Indian-only output filter. When RAG is enabled, non-Indian
scholarly papers are retained in the graph as well; this can increase extraction
calls. Defaults leave RAG disabled.
Abstracts are now retained in `PaperInfo` and structured output for retrieval.
Hook failures are logged without discarding extraction output. Reimport saved
records and run `index` to retry. Previously checkpointed papers are not silently
reprocessed; explicitly import old outputs. Older outputs may have titles only;
missing abstracts are not fabricated. Old outputs are also an Indian-positive
subset, so they do not support whole-conference coverage claims.

## Retrieval semantics

- Graph nodes cover organizations, geography, papers, paper-scoped authors,
  conference events, topics and unverified candidates. ORCID, when supplied, links
  authors across papers; identical names alone do not. Affiliation edges retain
  the paper, raw string, observed year and resolution method.
- Exact canonical/qualified alias matches resolve only when unique and trusted.
  Fuzzy matches are review suggestions. Historical/negated strings and recognized
  overseas-campus conflicts abstain. This is conservative entity resolution, not
  a general geographic classifier; the foreign-location guard is not exhaustive.
- IDs are stable across imports. DOI and URL identity take precedence over the
  fallback normalized-title hash. Renaming a CSV row should retain `entity_id`.
  Unknown organization observations never auto-promote to trusted facts.
- Graph and vector documents share one transactional database. Vectors retain
  model and content digests. Reingestion updates only changed text vectors and
  removes obsolete chunks. Searches reject stale indexes rather than silently
  omitting unindexed documents. Node changes and replacement paper observations retain before/after audit records;
  this is not a full event-sourced database or complete administrative audit API.
- Vector search uses normalized dense embeddings and exact dot products, with
  paper metadata/graph filtering before top-k and best-chunk deduplication. Optional
  `full_text` is split into overlapping 240-word chunks at 200-word strides.
- Graph search links explicit known entities and retrieves paper-backed relation
  evidence. Several named institutions mean a collaboration intersection. For
  repeated author names, results are alternatives, not an assertion of identity.
- Hybrid search enforces entity and metadata constraints, then uses reciprocal
  rank fusion (constant 60, configurable through the Python API) to combine graph
  and vector rankings. These are initial ranking defaults, not tuned evaluation
  weights. Arbitrary natural-language filters are not inferred; supply `--filters`.
- Supported filters: `country`, `state`, `organization_id`, `organization_type`,
  `author_id`, `author_name`, `topic`, `conference`, `year`, `year_min`, `year_max`. Organization
  predicates must match the same trusted organization. Geography comes from
  resolved institutions, not author names or emails. Aggregation counts distinct
  papers per institution across the filtered graph, not just retrieval top-k.
- `resolve AFFILIATION`, `subgraph ENTITY_ID --depth 2 --limit 100`, and
  `path SOURCE_ID TARGET_ID --depth 4` expose bounded graph operations. Paths can
  include geographic links and observations; a path is not itself evidence of
  collaboration. Use paper-backed graph search for collaboration questions.

Storage and exact vector scans target a local research corpus. Large deployments
will need indexed graph queries and an approximate-nearest-neighbor vector backend.
Unknown dates stay unknown; paper-specific affiliation does not assert a current
job or nationality. No citation relationships are invented without citation data.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python benchmark/test_rag.py
```

The second command checks real semantic paraphrase retrieval, graph collaboration,
hybrid filters, citations and incremental indexing on a synthetic corpus. Its
report is `benchmark/results/rag_smoke_report.json`; it is an integration check,
not a claim about retrieval quality on a held-out scientific benchmark.

`benchmark/results/existing_corpus_rag_report.json` also records a successful run
on 68 existing ACL 2025 and ACM SIGCOMM 2025 outputs. They have titles but no
stored abstracts. All three retrieval modes returned citations and a repeated
index operation was a no-op. That was the initial partial corpus check. The
database at `data/rag/sentry.sqlite3` (ignored by Git) is now backfilled from all
saved outputs using `rag.cli sync --index`; consult `data/rag/corpus_sync_report.json`
or `/v1/rag/status` for current coverage.
