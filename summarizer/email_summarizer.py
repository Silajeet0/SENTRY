"""
email_summarizer.py — the LLM call sites in the summarizer/ package: (1)
per-paper batch summarization, and (2) a conference-wide lead paragraph
synthesized from those per-paper summaries.

Deliberately a SEPARATE model/endpoint from LLM_MODEL (the main
orchestrator + extractors.llm_extractor model), configured via its own
SUMMARY_LLM_* env vars, falling back to LLM_* if unset so the feature works
out of the box even with only one model configured:

    SUMMARY_LLM_PROVIDER     (falls back to LLM_PROVIDER)
    SUMMARY_LLM_MODEL        (falls back to LLM_MODEL)
    SUMMARY_LLM_API_KEY      (falls back to LLM_API_KEY)
    SUMMARY_LLM_BASE_URL     (falls back to LLM_BASE_URL)
    SUMMARY_LLM_TEMPERATURE  (default 0.4 — see reasoning below)
    SUMMARY_BATCH_SIZE       (default 15 papers per LLM call)
    SUMMARY_MAX_ABSTRACT_CHARS (default 900 — per-abstract cap INSIDE the
                                 prompt; abstract_fetcher.py's 2500-char cap
                                 is a much looser upstream safety net, this
                                 is the tighter one that actually controls
                                 prompt-token budget across a whole batch)

WHY TEMPERATURE > 0 HERE (AND ONLY HERE):
    Every other LLM call in AEGIS (extractors.llm_extractor's per-paper
    metadata extraction, orchestrator.agent's tool-calling loop) runs at or
    near temperature 0 because the task is extraction/decision-making —
    there is a single correct answer (the paper's actual title, the actual
    next tool to call) and any variance is noise. Writing a fluent,
    readable one-to-two-sentence synthesis of an abstract is a genuinely
    different kind of task: temperature 0 tends toward stilted, repetitive
    phrasing across many papers in the same batch/run, which reads poorly
    in something a person is about to send as an email. A moderate
    temperature (default 0.4, not more) buys natural variety in sentence
    structure without inviting the model to invent facts not in the
    abstract — the SYSTEM_PROMPT below explicitly forbids that regardless.

WHY A SEPARATE MODEL INSTANCE (per the person's own local setup):
    On a 32GB M-series Mac running gpt-oss-20b locally as the main
    orchestrator/extraction model, that same process is also the one
    driving tool-calling for the agent loop — using it for a second,
    creative-writing-flavoured task at a different temperature at the same
    time is exactly the kind of "two jobs sharing one worker" contention
    orchestrator/runner.py already avoids for scraping (single job queue)
    for a different reason. Practically: a quantized 20B model (~12-13GB in
    Q4_K_M GGUF) plus a quantized 7-8B model (~4.5-5GB) both resident is
    comfortably under 32GB of unified memory, leaving headroom for
    Playwright/Chromium's own footprint during scraping. Point
    SUMMARY_LLM_BASE_URL at a second Ollama/LM Studio/llama.cpp server
    (can be the same host, different port) serving something like
    llama3.1:8b-instruct or qwen2.5:7b-instruct — either handles fluent
    short-form summarization comfortably and leaves the 20B model free for
    orchestration. If SUMMARY_LLM_* is left unset, this class transparently
    falls back to the main LLM_* config (same model, just a different
    temperature) so the feature still works with a single model configured
    — just without the concurrency benefit.

WHAT THE MODEL IS AND ISN'T TRUSTED WITH:
    Each batch call is given ONLY {index, title, abstract} per paper and
    asked to return {"summaries": [{"index", "summary"}]} — one sentence or
    two of plain-English synthesis, nothing else. It is never given (and
    the prompt explicitly tells it not to invent) author names,
    institutions, or the paper URL — those are spliced into the final email
    afterwards by build_email() straight from indian_papers_structured.json,
    which is the trusted ground truth for citations. A hallucinated
    one-sentence summary of an abstract the model *was* given is a much
    smaller, easier-to-spot failure mode than a hallucinated author name or
    link in what's about to be emailed out.

    The lead paragraph (write_intro_paragraph) is a SECOND, separate call,
    made once per email rather than once per batch, and is given even less:
    only the paper titles and the one-line summaries already produced by
    the first call above — never abstracts directly, never author/
    institution/link data. It's asked for 3-5 sentences of plain prose
    naming the paper count and conference/year and describing the general
    mix of topics, explicitly told not to invent anything beyond what's in
    those titles/summaries and not to restate per-paper detail (that's
    build_email()'s numbered list below it). If this call fails for any
    reason, build_email() falls back to a short templated sentence instead
    of leaving the email without a lead paragraph.
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """You are a research-communications assistant. You write clear, plain-English \
one-to-two sentence summaries of academic paper abstracts, for a research-management \
audience skimming a large batch of papers.

Rules:
- Base each summary ONLY on the abstract text given for that paper. Never invent \
results, numbers, or claims not present in the abstract.
- Do NOT mention author names, institutions, or affiliations in your summary — you \
are not given them, and they are handled separately.
- Plain English, no jargon dumps, no restating "This paper..." for every single one \
if you can vary the phrasing.
- Return ONLY valid JSON, no markdown fences, no commentary, in exactly this shape:
{"summaries": [{"index": 1, "summary": "..."}, {"index": 2, "summary": "..."}]}
Include exactly one entry per paper given below, in any order, using the same index."""

BATCH_USER_PROMPT_TEMPLATE = """Summarize each of the following {conference} {year} paper abstracts.

PAPERS:
{paper_blocks}

Return ONLY the JSON described in your instructions — one summary per paper index above."""

INTRO_SYSTEM_PROMPT = """You are a research-communications assistant. You write a longer, illustrative lead-in \
for an email that summarizes a batch of academic papers, based only on their titles and a one-line summary \
of each.

Rules:
- Base everything ONLY on the titles and summaries given below. Never invent details, subareas, or claims \
not evident from them.
- Open with 1-2 sentences stating the number of papers and the conference/year plainly.
- Then, in flowing prose (not a list, not bullet points, not numbered), briefly touch on EVERY paper's core \
idea or finding — a short clause or sentence each is enough, but every paper given below should get a \
mention, so the reader gets a quick preview of the whole set before reading the fuller numbered list further \
down. Group or connect related papers in the same sentence where it reads naturally (e.g. two papers on \
related topics can share a sentence), rather than mechanically restating "Paper N is about X" for every one.
- Aim for roughly 150-250 words for a typical batch (around 10-15 papers) — noticeably longer and more \
substantive than a short summary paragraph, since its job is to actually preview every paper's idea, not \
just name the general topic areas. For a larger batch, extend the length as needed so every paper still \
gets at least a brief mention — keep each individual mention to a short clause rather than a full sentence \
to manage the extra length, rather than dropping papers to stay within the 150-250 word target.
- Do NOT claim a specific track (e.g. "main track") unless it's explicitly stated below — you aren't given \
that information.
- Do NOT mention specific author names or institutions — those are handled in the numbered list right after \
this paragraph. You MAY reference a paper by a short recognizable phrase from its title if it helps the \
prose flow, but you don't need to quote titles in full.
- Return ONLY the paragraph(s) themselves as plain text — no heading, no markdown, no quotation marks, no \
commentary before or after it. One or two paragraphs is fine if that reads better than a single long block."""

INTRO_USER_PROMPT_TEMPLATE = """{conference} {year} — {count} paper(s) with Indian-affiliated authors.

PAPERS (title — one-line summary):
{paper_lines}

Write the introductory lead-in now — remember, every paper above should get at least a brief mention."""


class SummaryLLM:

    def __init__(self):
        self.provider = os.getenv("SUMMARY_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "nvidia")).lower()
        self.model = os.getenv("SUMMARY_LLM_MODEL", os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct"))
        self.api_key = os.getenv("SUMMARY_LLM_API_KEY", os.getenv("LLM_API_KEY", ""))
        self.base_url = os.getenv("SUMMARY_LLM_BASE_URL", os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
        self.temperature = float(os.getenv("SUMMARY_LLM_TEMPERATURE", "0.4"))
        self.batch_size = int(os.getenv("SUMMARY_BATCH_SIZE", "15"))
        self.max_abstract_chars = int(os.getenv("SUMMARY_MAX_ABSTRACT_CHARS", "900"))
        self.intro_max_tokens = int(os.getenv("SUMMARY_INTRO_MAX_TOKENS", "1200"))
        # Rate-limit resilience — a bigger conference means more batch calls
        # (and one intro call), each of which can independently collide with
        # a provider's per-minute limit. A small, unconditional pacing delay
        # before every call spreads them out instead of firing back-to-back,
        # which is what actually triggers a 429 in the first place; a higher
        # max_retries gives the SDK's own Retry-After-aware backoff more
        # chances to succeed on a busy/free-tier endpoint rather than giving
        # up after its (quite low) default of 2 retries.
        self.call_delay_seconds = float(os.getenv("SUMMARY_LLM_CALL_DELAY_SECONDS", "2"))
        self.max_retries = int(os.getenv("SUMMARY_LLM_MAX_RETRIES", "5"))
        # Set by write_intro_paragraph on failure so build_email() can report
        # WHY the lead paragraph fell back to the templated sentence, instead
        # of that fallback happening silently with no visible reason anywhere
        # — a real gap in the previous version, where a failed intro call
        # and "nothing to summarize" both looked identical from the outside.
        self.last_intro_error: Optional[str] = None

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=self.max_retries)

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str = SUMMARY_SYSTEM_PROMPT,
        json_mode: bool = True,
        max_tokens: int = 4000,
    ) -> str:
        if self.call_delay_seconds:
            time.sleep(self.call_delay_seconds)
        client = self._client()
        request = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        if json_mode and self._should_request_json_mode():
            request["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**request)
        except Exception:
            request.pop("response_format", None)
            response = client.chat.completions.create(**request)
        return (response.choices[0].message.content or "").strip()

    def _should_request_json_mode(self) -> bool:
        value = os.getenv("SUMMARY_LLM_RESPONSE_FORMAT_JSON", "auto").lower()
        if value in {"1", "true", "yes"}:
            return True
        if value in {"0", "false", "no"}:
            return False
        return "groq.com" in self.base_url or self.provider in {"openai", "groq"}

    @staticmethod
    def _parse_json(raw: str) -> dict:
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"No valid JSON found in summarizer LLM response: {raw[:200]}")

    def _summarize_one_batch(self, batch: list[dict], conference: str, year: str) -> dict[int, str]:
        """
        batch: list of {"index": int, "paper_title": str, "abstract": str}
        (only papers with a non-empty abstract should be passed in).
        Returns {index: summary}. Missing/failed indices simply aren't in
        the returned dict — the caller fills those in with a deterministic
        fallback rather than failing the whole batch.
        """
        blocks = []
        for item in batch:
            abstract = item["abstract"][: self.max_abstract_chars]
            blocks.append(f"[{item['index']}] Title: {item['paper_title']}\nAbstract: {abstract}")

        prompt = BATCH_USER_PROMPT_TEMPLATE.format(
            conference=conference, year=year, paper_blocks="\n\n".join(blocks)
        )

        try:
            raw = self._call_llm(prompt)
            parsed = self._parse_json(raw)
            out = {}
            for entry in parsed.get("summaries", []):
                idx = entry.get("index")
                summary = (entry.get("summary") or "").strip()
                if idx is not None and summary:
                    out[int(idx)] = summary
            return out
        except Exception as e:  # noqa: BLE001 — one bad batch must not kill the whole run
            log.warning(f"Summarizer batch call failed ({type(e).__name__}: {e}) — falling back to raw abstract excerpts for this batch")
            return {}

    def summarize_all(self, papers_with_abstracts: list[dict], conference: str, year: str) -> dict[int, str]:
        """
        papers_with_abstracts: list of merged paper dicts (paper_title,
        abstract, ...) as produced by abstract_fetcher.fetch_abstracts_for_papers,
        in the SAME order as the caller's numbering scheme (1-indexed).

        Returns {1-based index into papers_with_abstracts: summary text}.
        Papers with no usable abstract are skipped (not present in the
        result) — build_email() below falls back gracefully for those.
        """
        indexed = [
            {"index": i, "paper_title": p.get("paper_title", "Untitled"), "abstract": p["abstract"]}
            for i, p in enumerate(papers_with_abstracts, start=1)
            if p.get("abstract")
        ]

        summaries: dict[int, str] = {}
        for start in range(0, len(indexed), self.batch_size):
            batch = indexed[start:start + self.batch_size]
            summaries.update(self._summarize_one_batch(batch, conference, year))

        return summaries

    def write_intro_paragraph(
        self,
        papers_with_abstracts: list[dict],
        summaries_by_index: dict[int, str],
        conference: str,
        year: str,
    ) -> str:
        """
        A SECOND, separate LLM call — made once per email, not once per
        batch — that writes a short (3-5 sentence) lead paragraph
        synthesizing the whole set of papers, e.g. "Indian-affiliated
        authors had N papers at {conference} {year}. They spanned areas of
        X, Y, and Z. ...".

        Only ever given each included paper's title + its already-generated
        one-line summary (never the raw abstract, never author/institution/
        link data) — see the module docstring for why.

        Returns "" if there's nothing to summarize (no included papers) or
        if the LLM call fails for any reason — build_email() falls back to
        a short templated sentence in either case, so a broken lead
        paragraph never blocks the rest of the email from being generated.
        """
        lines = []
        for i, paper in enumerate(papers_with_abstracts, start=1):
            if not paper.get("abstract"):
                continue
            title = paper.get("paper_title", "Untitled")
            summary = summaries_by_index.get(i) or _fallback_summary(paper["abstract"])
            lines.append(f"- {title} — {summary}")

        if not lines:
            self.last_intro_error = None
            return ""

        prompt = INTRO_USER_PROMPT_TEMPLATE.format(
            conference=conference, year=year, count=len(lines), paper_lines="\n".join(lines)
        )

        try:
            text = self._call_llm(prompt, system_prompt=INTRO_SYSTEM_PROMPT, json_mode=False, max_tokens=self.intro_max_tokens)
            self.last_intro_error = None
            return text.strip().strip('"')
        except Exception as e:  # noqa: BLE001 — a failed intro call must not block the rest of the email
            self.last_intro_error = f"{type(e).__name__}: {e}"
            log.warning(f"Lead-paragraph call failed ({self.last_intro_error}) — falling back to a templated intro")
            return ""


# ---------------------------------------------------------------------------
# Deterministic email assembly — no LLM involved past this point. Title,
# authors, institutions, and the paper link all come straight from the
# trusted paper record (indian_papers_structured.json), never from the LLM.
# ---------------------------------------------------------------------------
def _fallback_summary(abstract: str) -> str:
    """Used only when the LLM summarizer produced nothing for this paper —
    a plain truncated excerpt of the real abstract beats silently omitting
    the paper or fabricating a summary."""
    excerpt = re.sub(r"\s+", " ", abstract).strip()
    if len(excerpt) <= 240:
        return excerpt
    return excerpt[:237].rsplit(" ", 1)[0] + "..."


def _format_authors(paper: dict) -> str:
    indian_authors = paper.get("authors_with_indian_affiliations") or []
    institutions = paper.get("indian_institutions") or []
    authors_str = ", ".join(indian_authors) if indian_authors else "Indian-affiliated author(s) not individually listed"
    if institutions:
        return f"{authors_str} ({'; '.join(institutions)})"
    return authors_str


def build_email(
    conference: str,
    year: str,
    papers_with_abstracts: list[dict],
    summaries_by_index: dict[int, str],
    intro_paragraph: str = None,
) -> dict:
    """
    Returns {"subject": str, "body": str, "paper_count": int,
    "papers_included": [...], "papers_skipped": [...]}.

    papers_with_abstracts is expected in the same order used to build
    summaries_by_index (1-based index = position in this list).

    intro_paragraph: pass an already-generated lead paragraph to use as-is
    (e.g. if a caller wants to review/edit it before assembly), or leave as
    None (the default) to have this function generate one itself via
    SummaryLLM.write_intro_paragraph — the LLM call happens right here so
    existing callers (summary_runner.py, run_conference_summary.py) get the
    new lead paragraph automatically with no changes on their end. Falls
    back to a short templated sentence if generation fails or there's
    nothing to summarize.
    """
    included = []
    skipped = []
    sections = []

    for i, paper in enumerate(papers_with_abstracts, start=1):
        title = paper.get("paper_title") or "Untitled"
        url = paper.get("paper_url", "")

        if not paper.get("abstract"):
            skipped.append({
                "paper_title": title,
                "paper_url": url,
                "reason": paper.get("error") or "No abstract could be retrieved",
            })
            continue

        summary = summaries_by_index.get(i) or _fallback_summary(paper["abstract"])
        entry_num = len(included) + 1
        sections.append(
            f"{entry_num}. {title}\n"
            f"   {summary}\n"
            f"   Indian author(s): {_format_authors(paper)}\n"
            f"   Link: {url}"
        )
        included.append({"paper_title": title, "paper_url": url})

    subject = f"Summary: Indian-Authored Papers at {conference} {year} ({len(included)} papers)"

    intro_fallback_reason = None
    if intro_paragraph is None and included:
        intro_llm = SummaryLLM()
        try:
            intro_paragraph = intro_llm.write_intro_paragraph(
                papers_with_abstracts, summaries_by_index, conference, year
            )
            if not intro_paragraph:
                intro_fallback_reason = intro_llm.last_intro_error or "Lead-paragraph call returned no text"
        except Exception as e:  # noqa: BLE001 — a broken lead paragraph must not break the whole email
            intro_fallback_reason = f"{type(e).__name__}: {e}"
            log.warning(f"Could not generate lead paragraph ({intro_fallback_reason}) — using a templated one instead")
            intro_paragraph = None

    intro_generated_by_llm = bool(intro_paragraph)

    if not intro_paragraph:
        intro_paragraph = (
            f"Indian-affiliated authors had {len(included)} paper(s) accepted at {conference} {year}."
            if included else
            f"No Indian-affiliated papers with a retrievable abstract were found for {conference} {year}."
        )

    intro = f"Hi,\n\n{intro_paragraph}\n"
    if skipped:
        intro += f"\n({len(skipped)} additional paper(s) are omitted below — their abstract could not be retrieved.)\n"

    body = intro + "\n" + "\n\n".join(sections) if sections else intro + "\nNo paper summaries could be generated."

    if skipped:
        skipped_lines = "\n".join(f"- {s['paper_title']} ({s['paper_url']})" for s in skipped)
        body += f"\n\nNot included above (abstract unavailable):\n{skipped_lines}"

    body += "\n\nBest regards,\n[Your name]"

    return {
        "subject": subject,
        "body": body,
        "paper_count": len(included),
        "papers_included": included,
        "papers_skipped": skipped,
        "intro_generated_by_llm": intro_generated_by_llm,
        "intro_fallback_reason": intro_fallback_reason,
        "generated_at": datetime.now().isoformat(),
    }
