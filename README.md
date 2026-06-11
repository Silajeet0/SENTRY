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
pip install requests beautifulsoup4 openai pdfminer.six python-dotenv
pip install playwright && playwright install chromium   # for Tier 2 (IEEE, ACM)
```

### 2. Configure LLM via environment variables

Create a `.env` file in the project root:

```env
LLM_PROVIDER=groq            # or: nvidia, openai, anthropic, ollama
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_KEY=your_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
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

Expand coverage by adding entries to `HTML_FRIENDLY_DOMAINS` (html_scraper), `BASE_URL_PATTERNS` (flat_link_extractor), and the `DOMAIN_INTERACTIONS` dict (browser_scraper).

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

## Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `nvidia` | Provider hint for JSON mode detection |
| `LLM_MODEL` | `meta/llama-3.3-70b-instruct` | Model name |
| `LLM_API_KEY` | _(empty)_ | API key |
| `LLM_BASE_URL` | NVIDIA NIM endpoint | Base URL of OpenAI-compatible API |
| `LLM_RESPONSE_FORMAT_JSON` | `auto` | Force JSON mode: `true`/`false`/`auto` |

---

## Form Filler (Standalone)

After running the pipeline, use the Selenium-based form filler to submit results to the IKDD data-sharing portal:

```bash
# Edit FORM_URL, CONFERENCE_NAME, YEAR, MONTH, VENUE inside the script
python Form_filler/run_selenium_filler.py
```

> **Note:** The `form_filling/` directory contains a legacy Agent-E-based form filler that requires a locally running Agent-E server. It is superseded by `Form_filler/` (Selenium) and is not maintained.

---

## Evaluation

Ground truth data for IEEE-ICDM 2025 is included under `data/ground_truth/`. The `data/eval_inputs/` directory contains link subsets used during development benchmarking.

---

## Known Issues & Limitations

- **`evaluation` module not included:** `pipeline.py` and `llm_extractor.py` import `from evaluation.india_rules import classify_affiliation`. This module must be present in the repo root for the pipeline to run. Ensure it is added before first use.
- IEEE Xplore pages behind institutional paywalls will fall through to Tier 4 (API). Affiliation coverage from Semantic Scholar may be incomplete for very recent papers.
- NeurIPS PDFs are page-1 only. Papers with affiliations stated only in the body will be missed by the prefilter and escalated to the LLM correctly.

---

## License

Research use. See `LICENSE` for terms.
