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

