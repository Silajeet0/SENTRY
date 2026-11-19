# SENTRY — Structured Extraction, Normalization, and Tiered Retrieval System for Scholarly Affiliation Tracking

A multi-tier pipeline for identifying Indian-affiliated authors across A/A\*-ranked CS conference proceedings. SENTRY scrapes and extracts paper metadata with a four-tier architecture and one LLM call per paper, then sends every extracted official affiliation to the fine-tuned Laya checkpoint for the binary `INDIAN`/`NOT_INDIAN` decision.

---

## Architecture Overview

```
              Proceedings URL
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Stage 1: Link Extraction      │
│  html_fetcher → flat / grouped extractor│
│  → links_raw/{conference}/{year}/       │
└───────────────────┬─────────────────────┘
                    │  links.json
                    ▼
┌─────────────────────────────────────────┐
│        Stage 2: Four-Tier Scraper       │
│  Tier 1 │ HTMLScraper   (requests+BS4)  │
│  Tier 2 │ BrowserScraper (Playwright)   │
│  Tier 3 │ PDFScraper    (pdfminer.six)  │
│  Tier 4 │ APIScraper    (SS/OA/CrossRef)│
└───────────────────┬─────────────────────┘
                    │  raw text content
                    ▼
┌─────────────────────────────────────────┐
│       Stage 3: LLM Extraction           │
│  LLMExtractor → structured JSON         │
│  paper metadata + official affiliations  │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│       Stage 4: Laya Classification      │
│  laya_finetuned_SENTRY → binary label   │
└───────────────────┬─────────────────────┘
                    │
                    ▼
        data/final_output/{conference}/{year}/
        ├── indian_papers_structured.json
        ├── processed_papers.json
        ├── errors.json
        └── summary.json
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt   
```

### 2. Configure LLM via environment variables

Create a `.env` file in the project root:

```env
LLM_PROVIDER=ollama           # or: nvidia, openai, anthropic, groq
LLM_MODEL=gpt-oss:20b   # used in the test run
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1 # or corresponding url

# Optional — only needed for the email-summary feature (see "Email Summary"
# below). Falls back to LLM_* above if unset, so the feature works with a
# single model too — but a SEPARATE, smaller model is recommended
SUMMARY_LLM_PROVIDER=ollama
SUMMARY_LLM_MODEL=llama3.1:8b-instruct
SUMMARY_LLM_API_KEY=ollama                       # placeholder — most local servers ignore this
SUMMARY_LLM_BASE_URL=http://localhost:11435/v1   # different port from the main model's server
SUMMARY_LLM_TEMPERATURE=0.4

# Tuning (all optional, falls back to defaults if not set)
SUMMARY_LLM_TEMPERATURE=0.4
SUMMARY_BATCH_SIZE=15
SUMMARY_MAX_ABSTRACT_CHARS=900
SUMMARY_INTRO_MAX_TOKENS=800
SUMMARY_LLM_CALL_DELAY_SECONDS=2
SUMMARY_LLM_MAX_RETRIES=5
SUMMARY_LLM_RESPONSE_FORMAT_JSON=auto
SUMMARY_LLM_REASONING_EFFORT=              # needs to be set accordingly if using a reasoning model

```

Any OpenAI-compatible endpoint works. The pipeline uses the `openai` Python SDK universally.

Production affiliation classification uses the local checkpoint at
`laya_finetuning/laya_finetuned_SENTRY`. Override it with
`SENTRY_LAYA_MODEL`, or set `SENTRY_LAYA_DEVICE` (`cpu`, `cuda`, or `auto`) and
`SENTRY_LAYA_THREADS` when needed. The extraction LLM returns paper metadata and
official affiliation strings; Laya receives each affiliation using the exact
`affiliation_geography` question used during fine-tuning.
Production retains Laya's probability argmax for diagnostics, then accepts
`INDIAN` only when `P(INDIAN) >= 0.9781`. Lower scores are conservatively mapped
to `NOT_INDIAN`; no deterministic fallback is used. The applied threshold and
raw argmax are retained on the validated decision for diagnostics.

AAAI OJS pages (`ojs.aaai.org`) are routed to the visible Playwright browser
tier because direct HTTP requests can return a bot challenge instead of paper
content. The browser waits up to 120 seconds for verification to clear; adjust
that limit with `SENTRY_BROWSER_CHALLENGE_TIMEOUT_SECONDS`.

### 3. Run the pipeline

```python
# run.py
from main_driver import run_pipeline

run_pipeline(
    proceeding_url="https://papers.nips.cc/paper_files/paper/2025",
    conference="NeurIPS",
    year="2025",
    max_papers=None,    # None = process all
    resume_from=0,
    delay=10            # seconds between papers
)
```

```bash
python run.py
```

### 4. Debug a single paper

```bash
# Run on a real URL
python run_single_paper.py https://ieeexplore.ieee.org/document/10825893

# Synthetic smoke test (no network required)
python run_single_paper.py --synthetic-indian
```

---

## Supported Conferences

| Structure | Conferences |
|-----------|-------------|
| **Flat** | NeurIPS, any IEEE Xplore proceedings |
| **Grouped (track-based)** | any aclanthology hosted proceedings, any ACM-DL hosted proceedings|
| **Grouped (two-level: volume → track)** | AAAI |
|**API BASED**|ICML, ICLR|

---

## Output Format

`indian_papers_structured.json` — array of objects:

```json
{
  "paper_number": 42,
  "paper_url": "https://ieeexplore.ieee.org/document/10825893",
  "paper_title": "...",
  "area_of_research": "Machine Learning",
  "total_authors": 4,
  "all_authors": [
    { "name": "Kshitij Jadhav", "affiliation": "IIT Bombay, Mumbai, India" }
  ],
  "authors_with_indian_affiliations": ["Kshitij Jadhav"],
  "indian_institutions": ["IIT Bombay, Mumbai, India"],
  "source": "browser",
  "processed_at": "2025-06-10T14:23:01"
}
```

`processed_papers.json` tracks every URL attempted (status: `indian_affiliated` / `no_indian_affiliation` / `error`) — used for crash-safe resume.

---

## Agentic Orchestrator

On top of the extraction and Laya classification pipeline, `orchestrator/` adds an LLM-driven
tool-calling layer that decides *which* conferences to run, *in what order*,
with *what track filters*, and *when to retry failures* — by calling the
same `main_driver` / `pipeline` functions you'd call by hand. The agent only
chooses which conference tools to call; affiliation geography is always decided
by the fine-tuned Laya checkpoint in the pipeline.

```bash
python orchestrator_cli.py
```

```
you> Extract Indian-affiliated papers from NeurIPS 2025, ICML 2025, and
     ACL 2025. Skip workshop tracks.
  [tool] resolve_conference_url(conference='NeurIPS', year='2025')
     -> {'proceeding_url': 'https://papers.nips.cc/paper_files/paper/2025', 'resolved': True, ...}
  [tool] detect_structure(conference='NeurIPS')
     -> ...

you> what's the status?
  [tool] get_run_status(conference='NeurIPS', year='2025')
     -> {'papers_attempted': 812, 'progress_pct': 14.2, 'status_breakdown': {...}}

you> retry the errors
  [tool] list_runs()
  [tool] retry_errors(conference='ACL', year='2025')
     -> {'status': 'started', 'retrying_count': 44, ...}
```

### Tools

| Tool | What it does |
|---|---|
| `resolve_conference_url(conference, year)` | Resolves a name+year to a proceedings URL from a known/pattern catalog. **Never guesses** ACM/IEEE URLs (per-instance DOIs/conhome IDs) — returns `resolved: false` and asks for the URL instead. |
| `validate_url(url)` | Lightweight reachability check before committing to a run. |
| `detect_structure(conference, proceeding_url)` | Reports `flat` vs `grouped` and which of `main_driver`'s three code paths (`openreview` / `acm_dl` / `generic`) will handle it. |
| `run_pipeline(conference, year, proceeding_url, skip_track_keywords, include_track_keywords, delay)` | Starts a run **in the background** and returns immediately — a full run over hundreds of papers can take hours. Track filtering (e.g. `skip_track_keywords=["workshop"]`) replaces the old interactive CLI prompt with keyword matching. |
| `get_run_status(conference, year)` | Polls the same `summary.json` / `errors.json` / `processed_papers.json` files the pipeline already saves after every paper — live progress, no separate tracking needed. |
| `retry_errors(conference, year)` | Re-runs only the papers that previously failed, in the background, using the exact links file from the original run. |
| `list_runs()` | Lists every run the orchestrator session knows about — used to resolve a vague "retry the errors" with no conference named. |
| `summarize_indian_authors(conference, year, refresh_cache)` | Reads the **already-extracted** `indian_papers_structured.json`, fetches each paper's abstract, and starts a **background** run that produces a cited email-body summary. See [Email Summary](#email-summary-indian-authors-work) below. |
| `get_summary_status(conference, year)` | Polls progress (`fetching_abstracts` → `summarizing` → `completed`) and returns the final `subject`/`body` once done. |
| `list_summary_runs()` | Lists every summary run + every conference/year on disk that's ready to summarize but hasn't been yet. |
| `initiate_form_filler(conference, year, month, venue, form_url, refresh_dedup_cache)` | Starts the RPA form-submission run in the background against the extracted, deduplicated papers for a conference/year. See [Form Filler](#form-filler-rpa) below. |
| `get_rpa_status(conference, year)` | Polls submission progress and returns the final per-paper submit/skip/fail breakdown once done. |

### Configuration

Same `.env` as the rest of SENTRY (`LLM_PROVIDER` / `LLM_MODEL` /
`LLM_API_KEY` / `LLM_BASE_URL`) — the orchestrator's LLM just needs to
support OpenAI-style function calling.

### Using main_driver.run_pipeline non-interactively yourself

```python
run_pipeline(
    proceeding_url="https://aclanthology.org/events/acl-2025/",
    conference="ACL",
    year="2025",
    interactive=False,                        # no blocking input() prompt
    track_skip_keywords=["workshop"],         # keyword-filtered track selection
    on_links_ready=lambda path: print(path),  # fires once links.json is resolved
)
```

---

## Talking to the Orchestrator from OpenWebUI

`orchestrator_api.py` wraps the same `Orchestrator` class the CLI uses
behind a minimal OpenAI-compatible `/v1/chat/completions` server, so a
self-hosted [OpenWebUI](https://github.com/open-webui/open-webui) instance
can drive SENTRY as a regular chat model — no browser extension, no
custom frontend code, just OpenWebUI's built-in "Connections" feature
pointed at a local server.

### 1. Start OpenWebUI (Docker)

```bash
docker run -d \
  --name open-webui \
  -p 3000:8080 \
  -v open-webui:/app/backend/data \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

- `-p 3000:8080` — OpenWebUI's own web UI will be reachable at
  `http://localhost:3000`.
- `-v open-webui:/app/backend/data` — persists OpenWebUI's own users/chats/
  settings in a named volume across container restarts.
- `--add-host=host.docker.internal:host-gateway` — lets the container
  reach services running on your host machine (the orchestrator API server
  below, plus Ollama if that's also on the host) via
  `host.docker.internal`, since `localhost` inside the container refers to
  the container itself, not your machine.

Wait a few seconds for the container to finish starting, then open
`http://localhost:3000` in a browser.

### 2. Start the SENTRY orchestrator API server

In a separate terminal, from the repo root, with your `.env` already
configured (see [Quick Start](#quick-start) above):

```bash
pip install fastapi uvicorn --break-system-packages
python orchestrator_api.py
```

This starts listening on `0.0.0.0:8091`. Leave it running — it needs to
stay up for the length of your OpenWebUI session, since it's the only
thing actually running SENTRY's pipeline, tools, and LLM calls.

### 3. Create your OpenWebUI account and connect SENTRY

1. On first visit to `http://localhost:3000`, OpenWebUI will prompt you to
   create a local admin account (this account only exists inside your own
   OpenWebUI container — nothing is sent externally).
2. Once logged in, go to **Settings → Admin Settings → Connections**.
3. Under **OpenAI API**, click **Add Connection** and fill in:
   - **Base URL**: `http://host.docker.internal:8091/v1`
     (if OpenWebUI is running on a *different* machine from the orchestrator
     API server, use that machine's LAN IP instead, e.g.
     `http://192.168.1.23:8091/v1`)
   - **API Key**: any placeholder value (e.g. `sentry`) — the server
     doesn't validate it.
4. Save the connection, then refresh the model picker in a new chat. A
   model named **`sentry-orchestrator`** should now appear alongside any
   other configured models.
5. Select `sentry-orchestrator` and start chatting — every message is
   forwarded to the same `Orchestrator.chat()` used by `orchestrator_cli.py`,
   so anything demonstrated in the [Agentic Orchestrator](#agentic-orchestrator)
   section above (starting runs, checking status, retrying errors,
   requesting summaries, triggering the form filler) works identically
   through the OpenWebUI chat window.

---

## Email Summary (Indian Authors' Work)

A downstream feature on top of an **already-completed** extraction run — it
reads `indian_papers_structured.json` rather than re-scraping or
re-classifying anything, so `run_pipeline` must have finished for that
conference/year first.

```bash
you> Summarize the works of the Indian authors in ICML 2025
  [tool] summarize_indian_authors(conference='ICML', year='2025')
     -> {'status': 'queued', 'message': 'Queued the email-summary run for ICML 2025 ...'}

you> is it done?
  [tool] get_summary_status(conference='ICML', year='2025')
     -> {'state': 'completed', 'stage': 'done', 'result': {'subject': 'Summary: Indian-Authored
        Papers at ICML 2025 (23 papers)', 'body': '...', 'paper_count': 23, ...}}
```

Or standalone, without the agent, the same role `run_single_paper.py` plays for extraction:

```bash
python run_conference_summary.py ICML 2025
python run_conference_summary.py ICML 2025 --max-papers 3   # quick smoke test
python run_conference_summary.py ICML 2025 --refresh-cache  # ignore cached abstracts, re-fetch all
```
### Configuration

```env
SUMMARY_LLM_PROVIDER=ollama
SUMMARY_LLM_MODEL=llama3.1:8b-instruct
SUMMARY_LLM_API_KEY=ollama
SUMMARY_LLM_BASE_URL=http://localhost:11435/v1   # different port from the main model's server
SUMMARY_LLM_TEMPERATURE=0.4    # default if unset
SUMMARY_BATCH_SIZE=15          # papers per LLM call, default if unset
SUMMARY_MAX_ABSTRACT_CHARS=900 # per-abstract cap inside the prompt, default if unset
```

All optional — every one falls back to a sensible default (or to the main
`LLM_*` config) if omitted.

### Output

`data/final_output/<conference>/<year>/email_summary.json`:

```json
{
  "subject": "Summary: Indian-Authored Papers at ICML 2025 (23 papers)",
  "body": "Hi,\n\nBelow is a summary of 23 paper(s) with Indian-affiliated authors accepted at ICML 2025.\n\n1. ...\n   <1-2 sentence summary>\n   Indian author(s): ... (...)\n   Link: https://openreview.net/forum?id=...\n\n...",
  "paper_count": 23,
  "papers_included": [ { "paper_title": "...", "paper_url": "..." } ],
  "papers_skipped": [ { "paper_title": "...", "paper_url": "...", "reason": "..." } ]
}
```

---

## Form Filler (RPA)

A downstream feature on top of an **already-completed** extraction run —
like the email summarizer, it reads `indian_papers_structured.json` for a
conference/year rather than re-scraping or re-classifying anything, and
submits each candidate paper as a nomination through a Selenium-driven
browser session against a target form (default: the IKDD data-sharing
portal, but any similarly-structured venue form works by pointing
`--form-url` elsewhere).

### Running it standalone

```bash
# Edit FORM_URL / IKDD_USERNAME / IKDD_PASSWORD in .env first (see below),
# then run against a conference/year that's already been extracted:
python Form_filler/run_selenium_filler.py \
  --conference NeurIPS \
  --year 2025 \
  --month Nov \
  --venue NeurIPS \
  --form-url https://ikdd.hosting.acm.org/ds-papers-form.php
```

- `--month` / `--venue` must match the **exact text** of the target form's
  dropdown options — these aren't free text fields, and a mismatch will
  fail the submission step rather than silently picking something close.
- `--no-refresh` skips re-scraping the venue's New/Approved lists and uses
  whatever dedup cache is already on disk (faster for repeated local
  testing, but can miss recently-approved papers).
- Programmatic use (e.g. from your own script, or the orchestrator's
  `initiate_form_filler` tool) is the same call via
  `run_form_filler(conference, year, month, venue, form_url,
  refresh_dedup_cache)`, returning a summary dict with
  `total_candidates`, `duplicates_skipped`, `submitted`, `failed`, and
  full per-paper detail lists for both duplicates and submission attempts.

### Configuration

```env
IKDD_USERNAME=your_login_email     # or the equivalent for whatever venue's
IKDD_PASSWORD=your_login_password  # New/Approved lists you're dedup-checking against
```

Required only for the dedup step's live refresh (step 2 above). If you'd
rather not scrape live, pre-populate a local cache with
`python -m utils.ikdd_dedup --refresh` once (interactively, with
credentials entered directly) and run the filler afterward with
`--no-refresh`.

### Via the orchestrator

```bash
you> Submit the ICML 2025 Indian-affiliated papers to IKDD for November, venue ICML
  [tool] initiate_form_filler(conference='ICML', year='2025', month='Nov', venue='ICML')
     -> {'status': 'started', ...}

you> how's the submission going?
  [tool] get_rpa_status(conference='ICML', year='2025')
     -> {'state': 'completed', 'total_candidates': 23, 'duplicates_skipped': 4,
        'submitted': 19, 'failed': 0, ...}
```

---

## Evaluation

Ground truth data for IEEE-ICDM 2025 is included under `data/ground_truth/`. The `data/eval_inputs/` directory contains link subsets used during development benchmarking.

---

## RAG and Laya evaluation extension

RAG queries are available through SENTRY chat and `/v1/rag/query`. The detailed
bootstrap, retrieval, benchmark, fine-tuning and institution-data instructions
are consolidated below. The conversational query guide remains at
[`rag/QUERYING.md`](rag/QUERYING.md).


## Institution bootstrap data
`Universities.csv` is the original 1,303-row UGC file supplied with this workspace;
it has not been edited. The importer uses its real column names, records the file
SHA-256 and row number, and labels its provenance `UGC_USER_SUPPLIED`. Its download
date and exact source URL were not supplied and are not invented.

`supplementary_institutions.csv` contains **32 manually normalized records**:

- All 23 IITs from the [government ISTI directory](https://www.indiascienceandtechnology.gov.in/organisations/centres-of-higher-learning/iits), including its second and third pages.
- [IISc](https://www.iisc.ac.in/about/general-information/contact/) and [ISI](https://asd.isical.ac.in/static/contact).
- Seven additional universities, completing coverage of the 12 notified institutions in the [UGC IoE list](https://ioe.ugc.ac.in/Home/ListofIOE) accessed on 2026-10-24. Four IITs and IISc already occur in the first two groups.

The CSV is a **review draft**, not an assertion that it has been human verified.
Every new record has `verification_status=NEEDS_REVIEW`. `iit_source_records.json`
preserves the names, state fields and website links extracted from the government
directory. Some of those website links are legacy URLs; they are preserved as
published. A blank address means the extractor did not capture one.

`ioe_status=NOTIFIED` means included in the referenced UGC list. A recommendation
or letter of intent is not treated as notification. `NOT_LISTED_IN_REFERENCED_IOE_LIST`
does not assert that an institute lacks other recognition. A blank notification
date means not transcribed. The 12 IoE records are IISc, IIT Bombay, IIT Delhi,
IIT Madras, IIT Kharagpur, Delhi University, University of Hyderabad, BHU, BITS
Pilani, MAHE, O P Jindal Global University and Shiv Nadar University.

Names and pipe-separated aliases were normalized editorially. City fields refer
to the primary institution/campus locality, not verified street addresses. Review
these fields against the per-row sources. ISI branches, overseas campuses, all
NITs, all IIITs and corporate labs are not an exhaustive part of this supplement.
In particular, an Indian parent organization does not make an overseas campus an
Indian affiliation. Bare IIT/NIT/ISI/VIT aliases are intentionally absent.

To use the draft without approving it:

```bash
.venv/bin/python -m rag.cli bootstrap KG_Data/Universities.csv KG_Data/supplementary_institutions.csv
```

Pending records are retained for inspection but excluded from trusted entity
resolution, organization vector search and geographic filters. If a pending
supplement duplicates a trusted UGC canonical name, its proposed aliases and
location are stored as a `REVIEW_RECORD`; they cannot alter trusted evidence.

After your review, copy the supplement to a reviewed CSV, correct any facts, and
set `verification_status=TRUSTED` only for approved rows. Import that file with the
same command and run `rag.cli index`. Keep `entity_id` unchanged across renames.
Reingest affected paper JSON to resolve previously unresolved affiliations.
Full candidate approval/merge tooling and automatic rewiring are deferred.

## Laya affiliation benchmark
The previously empty `benchmark_data.json` now contains 121 synthetic affiliation
cases: 40 positive, 40 negative and 41 ambiguous, plus 12 paper-level replay
fixtures. Every case has an ID, rationale, challenge category and fixed split
(24 validation / 97 test). Labels are editorial and need human review; this is
a diagnostic challenge set, not a representative scholarly sample or an
independent gold-standard dataset. Similar institutional names occur across
splits, so this split does not measure generalization to unseen institutions.

Positive means at least one **current institutional affiliation located in India**.
Negative requires identified foreign affiliations. Ambiguous means insufficient
evidence. Email, nationality, former jobs and a research subject mentioning India
do not establish location. Overseas campuses are classified by campus location.

## Run

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python benchmark/test_laya.py --backend both --model convaiinnovations/laya

# Reproduce the checkpoint used in the included report:
.venv/bin/python benchmark/test_laya.py --backend both --model convaiinnovations/laya \
  --revision 7b928d828b7b0e022f929d9bd2e44165aa270148 \
  --output benchmark/results/reproduction.json

# No model or download needed:
.venv/bin/python benchmark/test_laya.py --backend rules --output benchmark/results/rules_only.json
```

The script uses the [official Laya SDK](https://github.com/NandhaKishorM/laya) and
the requested [convaiinnovations/laya checkpoint](https://huggingface.co/convaiinnovations/laya),
with one typed choice question and three fixed criteria. It does not substitute
an LLM or silently fall back to rules. Missing dependencies, download errors and
invalid model responses produce a failed report and nonzero exit. The model is
loaded once; inference uses CPU with two threads by default. `--device`,
`--threads`, `--model`, `--revision`, `--dataset`, `--split` and `--output` are
configurable. Model files are cached under ignored `data/model_cache`.

The report records the dataset SHA-256, complete question/criteria, model revision,
library versions, per-case distributions and timings, confusion matrices,
per-class precision/recall/F1, macro-F1, Brier score and ECE. Brier is the sum
across three classes averaged over cases; ECE uses ten equal-width bins.
Probabilities are normalized only to compensate for SDK rounding. Reported
latency excludes model loading but includes the first inference call.
No confidence threshold was tuned on the test set.

## Independent benchmark comparison

`benchmark/benchmark_data.json` is kept separate from the fine-tuning corpus. Its
three-class labels are converted by
[`benchmark/parse_benchmark_data.py`](benchmark/parse_benchmark_data.py) into the exact binary Laya
format (`state`, `questions`, `expected`). `positive` projects to `INDIAN`, while
`negative` and `ambiguous` project to `NOT_INDIAN`. The original source label and
category remain in the comparison report for analysis.

Run both systems on all 121 independent cases with:

```bash
.venv/bin/python benchmark/test_independent_laya_vs_rules.py
```

The command writes the parsed input to
`benchmark/results/benchmark_data_laya.jsonl` and the comparison to
`benchmark/results/independent_laya_vs_rules_report.json`. It also saves overall,
per-class, confusion-matrix and category-accuracy PNG plots in
`benchmark/results/`. Use
`--split test` or `--split validation` to restrict the source split, and
`--max-cases N` for a short smoke run.

## Measured result

`results/laya_report.json` records an actual CPU run on 2026-10-24:

| Metric | Existing rules | Laya |
|---|---:|---:|
| All 121 cases: accuracy | 62.81% | 33.88% |
| All 121 cases: macro-F1 | 0.610 | 0.270 |
| Test split, 97 cases: accuracy | 62.89% | 34.02% |
| Positive recall | 95.00% | 42.50% |
| Ambiguous recall | 31.71% | 0.00% |
| False inclusions / false exclusions | 9 / 2 | 31 / 23 |
| Paper-level three-class accuracy, 12 fixtures | 58.33% | 25.00% |

Laya predicted 48 positives, 73 negatives and **zero ambiguous** cases. Its mean
CPU latency was approximately 2.27 seconds per case, excluding loading. This
historical report evaluates the original three-class base-checkpoint experiment;
it is not the production checkpoint result. The current production pipeline uses
`laya_finetuning/laya_finetuned_SENTRY` and the binary contract.

During loading, the SDK warned that an unrelated 11+-option temperature entry
was clamped to its supported range. This benchmark uses three options. Confidence
still has no domain-specific calibration guarantee; the measured Brier/ECE are
diagnostics, not proof of calibration.

The paper replay groups structured author-affiliation decisions and applies the
pipeline's positive-only inclusion policy. It does **not** rerun web acquisition,
author extraction or form filling and is not a full end-to-end pipeline accuracy
measurement. The separate
`test_rag.py` script tests retrieval integration on synthetic paper fixtures.

For the Laya fine-tuning corpus, see
[Laya fine-tuning data](#laya-fine-tuning-data). It contains 1,200
binary JSONL examples (400 `INDIAN`, 800 `NOT_INDIAN`) in Laya's
`{state, questions, expected}` format, plus reproducible
train/validation/test splits.

The following comparison scripts are retained as historical diagnostics for the
former deterministic classifier. They are not imported by the production
pipeline and can be removed together with `evaluation/india_rules.py` when that
historical comparison is no longer needed. To reproduce the old comparison on
the untouched test split, run:

```bash
.venv/bin/python benchmark/test_laya_vs_rules.py
```

The default inputs are `laya_finetuning/finetuning_data/test.jsonl` and
`laya_finetuning/laya_finetuned_SENTRY`. The script writes
`benchmark/results/laya_vs_rules_test_report.json`. It maps the
rule classifier's `positive` result to `INDIAN` and both `negative` and
`ambiguous` results to `NOT_INDIAN`, while retaining the raw rule label in the
report.

For the Laya fine-tuning corpus, see
[Laya fine-tuning data](#laya-fine-tuning-data). It contains 1,200
binary JSONL examples (400 `INDIAN`, 800 `NOT_INDIAN`) in Laya's
`{state, questions, expected}` format, plus reproducible
train/validation/test splits.

## Laya fine-tuning data
This directory contains a reproducible, synthetic supervised corpus for adapting
Laya to SENTRY's binary affiliation decision:

- `INDIAN`: sufficient evidence that at least one current institutional
  affiliation is located in India;
- `NOT_INDIAN`: the affiliation is outside India or there is not enough reliable
  evidence to establish a current Indian affiliation.

There are **1,200 examples**: 400 `INDIAN` followed by 800 `NOT_INDIAN` rows. The
seven requested positive categories, seven foreign-negative categories and eight
former ambiguous categories are all represented. The files contain 960 training,
120 validation and 120 test rows, with 320/640, 40/80 and 40/80
`INDIAN`/`NOT_INDIAN` examples respectively. `all.jsonl` is the complete corpus.

The source templates originally carried three labels. Every former `ambiguous`
row is now a final `NOT_INDIAN` row because it does not establish a current Indian
affiliation.

Each JSONL row is directly compatible with Laya's native training/evaluation
reader:

```json
{"state":"IIT Bombay, Mumbai","questions":{"affiliation_geography":{"type":"choice","instructions":"Does this affiliation provide sufficient evidence that the current institution or campus is located in India?","criteria":{"INDIAN":"Sufficient evidence that the current institution or campus is in India.","NOT_INDIAN":"No sufficient evidence that the current institution or campus is in India."}}},"expected":{"affiliation_geography":"INDIAN"}}
```

Each row has one `expected` target field. Exact counts, hashes and policy are in
[laya_finetuning/finetuning_data/manifest.json](laya_finetuning/finetuning_data/manifest.json).

The generator is [benchmark/generate_laya_finetuning.py](benchmark/generate_laya_finetuning.py).
Re-running it with the same code and seed (`20261029`) reproduces the same rows
and split assignments.

## Validate and train

```bash
.venv/bin/python benchmark/generate_laya_finetuning.py
.venv/bin/python - <<'PY'
from laya.evals import Dataset
for name in ("train", "validation", "test"):
    data = Dataset.from_jsonl(f"benchmark/laya_finetuning/{name}.jsonl")
    print(name, len(data))
PY

.venv/bin/laya-train \
  --data benchmark/laya_finetuning/train.jsonl \
  --eval benchmark/laya_finetuning/validation.jsonl \
  --base convaiinnovations/laya \
  --out data/models/laya-india-affiliation \
  --epochs 4 --micro-batch 8 --grad-accum 8 --loss rlcd --seed 20261029
```

On an offline machine, point `--base` at a downloaded local Laya checkpoint.
The command needs the Laya training dependencies and a writable output path.
Keep the test split untouched while choosing epochs, learning rates, or
abstention thresholds. Evaluate the resulting checkpoint with
`benchmark/test_laya.py --model data/models/laya-india-affiliation --split test`.

## Labeling cautions

This is generated training data, not an externally verified gold-standard
corpus. It uses manually written institution families and controlled variations.
The same families can appear across splits, so the test score measures coverage
of these policy patterns rather than generalization to unseen institutions.
The 1:2 `INDIAN`/`NOT_INDIAN` ratio is a valid choice when unresolved evidence is
operationally treated as negative; it is intentionally not a 50/50 class-balanced
corpus. Report per-class recall, macro-F1, balanced accuracy and the confusion
matrix in addition to accuracy. If deployment prevalence differs materially,
consider class weights or calibration during model selection rather than judging
the model by accuracy alone.
Review labels before training. In particular:

- A current Indian affiliation makes a multiple-affiliation example `INDIAN`;
  a foreign coauthor does not cancel it.
- A former Indian job does not make a current foreign affiliation `INDIAN`.
- A bare acronym, email address, city alone, India as a research topic, or any
  other unresolved evidence is `NOT_INDIAN` under the binary policy.
- An overseas campus is classified by its current campus location.
- `IIT`, `NIT`, `ISI`, `VIT` and similar acronyms are `NOT_INDIAN` without a
  disambiguating campus or institution.

For a serious model release, add reviewed examples from real affiliation strings,
hold out institution families by name, and record annotator agreement. Do not
use these synthetic labels alone to make employment, nationality or eligibility
decisions.

## SENTRY RAG extension
For conversational queries, HTTP endpoints and complete-corpus backfilling,
see [rag/QUERYING.md](rag/QUERYING.md). RAG is now connected to SENTRY's orchestrator.

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
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m rag.cli bootstrap KG_Data/Universities.csv KG_Data/supplementary_institutions.csv
```

The SQLite database defaults to `data/rag/sentry.sqlite3`; override with the global
`--db` option. New supplement records remain pending until you review them; see
the [Institution bootstrap data](#institution-bootstrap-data) section. No database server or hosted vector service is needed.

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
