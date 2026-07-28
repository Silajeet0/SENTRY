# AEGIS — Academic Extraction & Geolocation Intelligence System

A multi-tier, deterministic pipeline for identifying Indian-affiliated authors across A/A\*-ranked CS conference proceedings. AEGIS scrapes, extracts, and classifies paper metadata using a four-tier scraping architecture followed by a single LLM call per paper — no agent loops, no planning steps.

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

## Repository Structure

```
AEGIS_Refactored/
├── run.py                          # Main entrypoint
├── run_single_paper.py             # Single-paper debug / smoke-test entrypoint
├── main_driver.py                  # Orchestrates Stage 1 → Stage 2 → Stage 3
├── pipeline.py                     # Core per-paper processing + run_pipeline()
│
├── scrapers/
│   ├── base.py                     # BaseScraper ABC + ScrapeResult dataclass
│   ├── html_scraper.py             # Tier 1: requests + BeautifulSoup
│   ├── browser_scraper.py          # Tier 2: Playwright + IEEE author page logic
│   ├── pdf_scraper.py              # Tier 3: pdfminer.six, NeurIPS HTML fallback
│   └── api_scraper.py              # Tier 4: Semantic Scholar, OpenAlex, CrossRef
│
├── extractors/
│   └── llm_extractor.py            # OpenAI-compatible LLM call + JSON parsing
│
├── workflows/
│   ├── html_fetcher.py             # Playwright-based proceedings page fetcher
│   ├── track_detector.py           # flat vs. grouped structure detection
│   └── link_extractors/
│       ├── flat_link_extractor.py  # IEEE, NeurIPS link extraction
│       ├── grouped_link_extractor.py  # ACL, ACM track-grouped extraction
│       └── skip_patterns.py        # Non-paper title filter
│
├── utils/
│   ├── ieee_author_parser.py       # IEEE author/affiliation text parsing helpers
│   └── track_selector_cli.py       # Interactive CLI for ACL/ACM track selection
│
├── Form_filler/                    # Selenium-based form auto-filler (standalone)
│   ├── selenium_filler.py
│   └── run_selenium_filler.py
│
├── data/
│   ├── html/                       # Cached proceedings HTML pages
│   ├── links_raw/                  # Extracted paper links per conference/year
│   ├── final_output/               # Indian-affiliated paper results
│   ├── eval_inputs/                # Evaluation subsets
│   └── ground_truth/              # Labelled ground truth for eval
│
└── requirements.txt
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
LLM_PROVIDER=groq            # or: nvidia, openai, anthropic, ollama
LLM_MODEL=openai/gpt-oss-20b   # used in the test run
LLM_API_KEY=your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1 # or corresponding url

# Optional — only needed for the email-summary feature (see "Email Summary"
# below). Falls back to LLM_* above if unset, so the feature works with a
# single model too — but a SEPARATE, smaller model is recommended: the
# summarizer runs at temperature > 0 (fluent prose, not extraction) and
# benefits from its own instance rather than sharing the 20B model that's
# also driving the orchestrator's tool-calling loop. On a 32GB Mac running
# gpt-oss-20b locally (~12-13GB quantized), a second local 7-8B model
# (~4.5-5GB quantized, e.g. llama3.1:8b-instruct or qwen2.5:7b-instruct via
# a second Ollama/LM Studio/llama.cpp server) both fit comfortably in
# unified memory with room to spare for Playwright/Chromium during scraping.
SUMMARY_LLM_PROVIDER=ollama
SUMMARY_LLM_MODEL=llama3.1:8b-instruct
SUMMARY_LLM_API_KEY=ollama                       # placeholder — most local servers ignore this
SUMMARY_LLM_BASE_URL=http://localhost:11435/v1   # different port from the main model's server
SUMMARY_LLM_TEMPERATURE=0.4
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
| **Flat** | NeurIPS / NIPS, IEEE-ICDM, IEEE-CVPR, any IEEE Xplore proceedings |
| **Grouped (track-based)** | ACL, EMNLP, NAACL, ACM-KDD, ACM-SIGMOD, ACM-IKDD |

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
  ⚙️  resolve_conference_url(conference='NeurIPS', year='2025')
     → {'proceeding_url': 'https://papers.nips.cc/paper_files/paper/2025', 'resolved': True, ...}
  ⚙️  detect_structure(conference='NeurIPS')
     → {'structure': 'flat', 'handler': 'generic_flat', ...}
  ⚙️  run_pipeline(conference='NeurIPS', year='2025', proceeding_url='...')
     → {'status': 'started', ...}
  ...

you> what's the status?
  ⚙️  get_run_status(conference='NeurIPS', year='2025')
     → {'papers_attempted': 812, 'progress_pct': 14.2, 'status_breakdown': {...}}

you> retry the errors
  ⚙️  list_runs()
  ⚙️  retry_errors(conference='ACL', year='2025')
     → {'status': 'started', 'retrying_count': 44, ...}
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

### How retries actually work

`pipeline.run_pipeline` already tracks every attempted URL in
`processed_papers.json` (crash-safe resume). `retry_errors` backs that file
up, strips out just the entries with `status: "error"`, and calls
`pipeline.run_pipeline` again against the same links file — successful
papers stay skipped, only the previously-failed ones get reprocessed.

### Configuration

Same `.env` as the rest of AEGIS (`LLM_PROVIDER` / `LLM_MODEL` /
`LLM_API_KEY` / `LLM_BASE_URL`) — the orchestrator's LLM just needs to
support OpenAI-style function calling (Groq's `openai/gpt-oss-20b` does).

### Using main_driver.run_pipeline non-interactively yourself

`main_driver.run_pipeline` grew three new optional parameters, all
backward-compatible (default `interactive=True` preserves the exact old
CLI-prompt behavior for `run.py`):

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

## Email Summary (Indian Authors' Work)

A downstream feature on top of an **already-completed** extraction run — it
reads `indian_papers_structured.json` rather than re-scraping or
re-classifying anything, so `run_pipeline` must have finished for that
conference/year first.

```bash
you> Summarize the works of the Indian authors in ICML 2025
  ⚙️  summarize_indian_authors(conference='ICML', year='2025')
     → {'status': 'queued', 'message': 'Queued the email-summary run for ICML 2025 ...'}

you> is it done?
  ⚙️  get_summary_status(conference='ICML', year='2025')
     → {'state': 'completed', 'stage': 'done', 'result': {'subject': 'Summary: Indian-Authored
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

Every other LLM call in AEGIS (per-paper metadata extraction, the
orchestrator's tool-calling loop) runs at or near temperature 0 — an
extraction/decision task has one correct answer, so any variance is noise.
Writing a readable summary of an abstract is a different kind of task:
temperature 0 tends toward stilted, repetitive phrasing across a whole
batch of papers, which reads poorly in something about to be sent as an
email. This feature is the **one place** in AEGIS that intentionally runs
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

## Form Filler (Standalone)

After running the pipeline, use the Selenium-based form filler to submit results to the IKDD data-sharing portal:

```bash
# Edit FORM_URL, CONFERENCE_NAME, YEAR, MONTH, VENUE inside the script
python Form_filler/run_selenium_filler.py
```
---

## Evaluation

Ground truth data for IEEE-ICDM 2025 is included under `data/ground_truth/`. The `data/eval_inputs/` directory contains link subsets used during development benchmarking.

---

