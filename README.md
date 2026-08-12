# SENTRY — Structured Extraction, Normalization, and Tiered Retrieval System for Scholarly Affiliation Tracking

A multi-tier, deterministic pipeline for identifying Indian-affiliated authors across A/A\*-ranked CS conference proceedings. SENTRY scrapes, extracts, and classifies paper metadata using a four-tier scraping architecture followed by a single LLM call per paper — no agent loops, no planning steps.

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
│  + post-hoc India keyword verifier      │
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
| **Grouped (track-based)** | ACL, EMNLP, NAACL, any ACM-DL hosted proceedings|
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

On top of the deterministic pipeline above, `orchestrator/` adds an LLM-driven
tool-calling layer that decides *which* conferences to run, *in what order*,
with *what track filters*, and *when to retry failures* — by calling the
same `main_driver` / `pipeline` functions you'd call by hand. Nothing about
the scraping/extraction pipeline itself became less deterministic: the agent
only ever chooses which deterministic tool to call next.

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
support OpenAI-style function calling (Groq's `openai/gpt-oss-20b` does).

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

### What it does, per paper

1. **Get the abstract.** For OpenReview-hosted conferences (ICML, ICLR, ...
   — detected from the URL, same check `run_single_paper.py` already uses)
   the abstract is already a field on the OpenReview note, so it's fetched
   directly via the API — no scraping at all. For everything else
   (ACL/NeurIPS PDFs, IEEE/ACM pages, ...) it reuses the **exact same
   four-tier scraper** `run_pipeline` uses (`pipeline.TIERS` — same
   persistent Playwright browser, same ACM cookie jar), then pulls just the
   Abstract section out of the scraped text with a heading-anchored regex
   (falling back to a leading text window if no heading is found).
   Title/authors/institutions are **not** re-derived from scraped text —
   they're already trustworthy in `indian_papers_structured.json`.
2. **Cache it.** Every fetched abstract is saved to
   `data/final_output/<conference>/<year>/abstracts_cache.json`
   immediately, crash-safe and resumable like `processed_papers.json`.
   Asking for the same conference/year's summary again later reuses the
   cache rather than re-scraping (abstracts don't change once published);
   pass `refresh_cache=True` to force a re-fetch.
3. **Summarize it.** Once every paper's abstract is in hand, they're
   batched (default 15 papers/call — override with `SUMMARY_BATCH_SIZE`)
   to a **separate summarization LLM** and written into a 1-2 sentence
   plain-English synthesis each. This is the only LLM call in the whole
   feature.
4. **Assemble the email — deterministically.** The LLM is only ever given
   `{title, abstract}` and asked to return `{summary}` per paper — it is
   never given, and is explicitly told not to invent, author names,
   institutions, or the paper link. Those are spliced into the final email
   afterwards straight from `indian_papers_structured.json`, so the
   citation the recipient actually sees can't contain a hallucinated name
   or link even if the prose summary of some abstract is imperfect. Papers
   whose abstract couldn't be retrieved are listed separately at the bottom
   ("Not included above") rather than silently dropped or faked.

### Why a separate model, at a different temperature

Every other LLM call in SENTRY (per-paper metadata extraction, the
orchestrator's tool-calling loop) runs at or near temperature 0 — an
extraction/decision task has one correct answer, so any variance is noise.
Writing a readable summary of an abstract is a different kind of task:
temperature 0 tends toward stilted, repetitive phrasing across a whole
batch of papers, which reads poorly in something about to be sent as an
email. This feature is the **one place** in SENTRY that intentionally runs
above temperature 0 (default 0.4).

Because it's a different kind of workload, it's also configured as a
**separate model/endpoint** from the main orchestrator/extraction model —
recommended, not required (it falls back to the main `LLM_*` config if
`SUMMARY_LLM_*` is unset). On the 32GB M-series Mac setup this is built
for, running `gpt-oss-20b` as the main model: a quantized 20B model
(~12-13GB) and a quantized 7-8B model (~4.5-5GB, e.g.
`llama3.1:8b-instruct` or `qwen2.5:7b-instruct` served locally via a second
Ollama/LM Studio/llama.cpp instance on a different port) both fit
comfortably in unified memory at once, with headroom left for
Playwright/Chromium during the scraping stage. This also means the 20B
model stays free to keep driving the orchestrator's tool-calling loop while
the 8B model handles summarization, rather than one process serializing
both jobs on the same weights.

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

### What it does, step by step

1. **Load the extracted candidates.** Reads
   `data/final_output/<conference>/<year>/indian_papers_structured.json`.
   If that file doesn't exist yet, it errors out with a clear message
   pointing back at running the extraction pipeline first.
2. **Deduplicate against what's already submitted.** Before submitting
   anything, `utils/ikdd_dedup.py` scrapes (or reuses a cached copy of) the
   target venue's **New** and **Approved** nomination lists, then checks
   every candidate paper's title against both using token-based Jaccard
   similarity (threshold `0.85`, over normalized — lowercased,
   accent- and punctuation-stripped — titles). Anything scoring above the
   threshold against an existing entry is treated as already-present and
   skipped, along with the score and the specific existing title it
   matched, so you can audit a near-miss instead of just trusting the
   skip.
3. **Submit only the genuinely new candidates.** For each remaining paper,
   `Form_filler/selenium_filler.py` drives a real (non-headless) browser
   session: it navigates to the form, fills in the standard fields
   (title, conference, month, venue, and the paper's Indian author/
   institution details pulled straight from the already-verified JSON —
   nothing is re-typed by an LLM at this stage), and handles the
   "Others"-category dropdown's dynamic follow-up text field when a
   paper's `area_of_research` doesn't match one of the form's fixed
   options.
4. **Run as a single sequential job.** Submissions happen one at a time in
   the same browser session, deliberately not in parallel — a second
   concurrent session hitting the same form would itself look like
   automated/bot traffic to the target site, independent of how fast any
   individual submission is.

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