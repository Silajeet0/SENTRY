"""
summarizer/ — a downstream, independent feature on top of AEGIS's existing
extraction output.

Takes an ALREADY-extracted conference/year (data/final_output/<conference>/
<year>/indian_papers_structured.json — the same file initiate_form_filler
reads) and produces a concise, cited email-body summary of the work done by
its Indian-affiliated authors.

Two stages, same "deterministic pipeline + one designated LLM step" ethos as
the rest of AEGIS:

    1. abstract_fetcher.py  — deterministic. Gets each paper's abstract via
       the OpenReview API fast path (ICML/ICLR/... — no scraping needed, the
       abstract is already on the note) or AEGIS's existing four-tier
       scraper (everything else), then extracts just the Abstract window
       with a regex heuristic. No LLM call here. Title/authors/institutions
       are NOT re-derived — they're already trustworthy in
       indian_papers_structured.json from the main pipeline's extraction
       LLM call, so pulling them from scraped text again would only add a
       chance to disagree with the ground truth already on disk.

    2. email_summarizer.py — the ONE LLM call site in this package, made in
       batches against a SEPARATE model/endpoint from the main
       orchestrator/extraction LLM (see SUMMARY_LLM_* env vars), at
       temperature > 0 (this is a writing task, not extraction — some
       lexical variety is desirable, unlike the rest of AEGIS which runs at
       temperature 0). The LLM is only ever asked to write a 1-2 sentence
       plain-English synthesis of one paper's abstract at a time (returned
       as JSON keyed by paper index) — it never sees or invents the
       title/authors/institutions/link that go in the final citation. Those
       are spliced in afterwards by plain Python from the trusted structured
       data, so a hallucinated author name or link can't slip into the
       citation the recipient sees.
"""
