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

# Ollama-native JSON schemA mirroring
# the shape described in SUMMARY_SYSTEM_PROMPT. Passing this constrains
# decoding via GBNF grammar so the model literally cannot emit a token that
# breaks the schema
SUMMARIES_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "summary": {"type": "string"},
                },
                "required": ["index", "summary"],
            },
        }
    },
    "required": ["summaries"],
}

BATCH_USER_PROMPT_TEMPLATE = """Summarize each of the following {conference} {year} paper abstracts.

PAPERS:
{paper_blocks}

Return ONLY the JSON described in your instructions — one summary per paper index above."""

INTRO_SYSTEM_PROMPT = """You are a research-communications assistant. You write a short, thematic lead-in \
for an email that summarizes a batch of academic papers, based only on their titles and a one-line summary \
of each.

Your job is NOT to narrate each paper one by one — that reads like a list stitched into prose, which is \
exactly what to avoid. Instead:
- Read across ALL the papers given below and identify the small number of overarching research areas or \
themes they cluster into (e.g. "bandit problems", "diffusion models", "fairness in selection", "online \
convex optimization") — usually somewhere between 3 and 7 themes depending on how many papers there are.
- Open with 1 sentence stating the number of papers and the conference/year plainly.
- Then, in 1-3 further sentences, name the thematic areas the papers span — a concise, flowing "they spanned \
the areas of X, Y, and Z" style sentence (or two), not a paper-by-paper walkthrough.
- Optionally close with one more sentence noting any genuine, evident overall pattern across the set (e.g. \
"much of the work leans theoretical" or "several papers focus on efficiency/scaling") — ONLY if it's actually \
evident from the titles/summaries given, never invented.
- Base everything ONLY on the titles and summaries given below. Never invent papers, subareas, or claims not \
evident from them, and never claim a specific track (e.g. "main track") unless it's explicitly stated below.
- Keep the WHOLE thing concise — roughly 4-7 sentences total, regardless of how many papers are in the batch. \
More papers should mean identifying broader/coarser themes to stay concise, not covering more individual \
papers.
- Do NOT mention specific author names, institutions, or individual paper titles — save that detail for the \
numbered list that follows this paragraph.
- Return ONLY the paragraph itself as plain text — no heading, no markdown, no quotation marks, no commentary \
before or after it."""

INTRO_USER_PROMPT_TEMPLATE = """{conference} {year} — {count} paper(s) with Indian-affiliated authors.

PAPERS (title — one-line summary):
{paper_lines}

Identify the overarching themes across these papers and write the thematic lead-in now — a concise synthesis \
of the areas covered, not a walkthrough of each paper."""


class SummaryLLM:

    def __init__(self):
        self.provider = os.getenv("SUMMARY_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "nvidia")).lower()
        self.model = os.getenv("SUMMARY_LLM_MODEL", os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct"))
        self.api_key = os.getenv("SUMMARY_LLM_API_KEY", os.getenv("LLM_API_KEY", ""))
        self.base_url = os.getenv("SUMMARY_LLM_BASE_URL", os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
        self.temperature = float(os.getenv("SUMMARY_LLM_TEMPERATURE", "0.4"))
        self.batch_size = int(os.getenv("SUMMARY_BATCH_SIZE", "15"))
        self.max_abstract_chars = int(os.getenv("SUMMARY_MAX_ABSTRACT_CHARS", "900"))   
        self.intro_max_tokens = int(os.getenv("SUMMARY_INTRO_MAX_TOKENS", "800"))
        self.reasoning_effort = os.getenv("SUMMARY_LLM_REASONING_EFFORT", "").strip().lower()
        self.call_delay_seconds = float(os.getenv("SUMMARY_LLM_CALL_DELAY_SECONDS", "2"))
        self.max_retries = int(os.getenv("SUMMARY_LLM_MAX_RETRIES", "5"))
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
        json_schema: Optional[dict] = None,
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
        extra_body = {}
        if json_mode and self._should_request_json_mode():
            request["response_format"] = {"type": "json_object"}
            if json_schema and self.provider == "ollama":
                extra_body["format"] = json_schema
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        try:
            response = client.chat.completions.create(**request, extra_body=extra_body or None)
        except Exception:
            request.pop("response_format", None)
            response = client.chat.completions.create(**request)

        choice = response.choices[0]
        text = (choice.message.content or "").strip()

        if not text:

            finish_reason = getattr(choice, "finish_reason", "unknown")
            usage = getattr(response, "usage", None)
            completion_tokens = getattr(usage, "completion_tokens", "unknown") if usage else "unknown"
            if finish_reason == "length":
                raise RuntimeError(
                    f"Model returned empty content with finish_reason='length' "
                    f"(completion_tokens={completion_tokens}, max_tokens requested={max_tokens}) "
                    f"— this is the signature of a reasoning model spending its entire token "
                    f"budget on internal reasoning before writing the visible answer. Raise "
                    f"max_tokens for this call, or set SUMMARY_LLM_REASONING_EFFORT=low if "
                    f"you're using a gpt-oss-style model."
                )
            raise RuntimeError(
                f"Model returned empty content (finish_reason={finish_reason!r}, "
                f"completion_tokens={completion_tokens})."
            )

        return text

    def _should_request_json_mode(self) -> bool:
        value = os.getenv("SUMMARY_LLM_RESPONSE_FORMAT_JSON", "auto").lower()
        if value in {"1", "true", "yes"}:
            return True
        if value in {"0", "false", "no"}:
            return False

        return (
            "groq.com" in self.base_url
            or self.provider in {"openai", "groq", "ollama"}
            or any(host in self.base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))
        )

    @staticmethod
    def _extract_json_span(raw: str) -> Optional[str]:
        """
        Find the first balanced {...} span, respecting string/escape state,
        instead of the old `re.search(r"\\{.*\\}", raw, re.DOTALL)`.
        """
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]
        return None  

    @staticmethod
    def _repair_json(candidate: str) -> str:
        """
        Cheap, deterministic fixes for the handful of malformations small
        local models actually produce, tried only after a straight
        json.loads() has already failed.
        """
        # Trailing commas before a closing bracket: {"a": 1,} / [1, 2,]
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        # Smart/curly quotes some models substitute for straight quotes.
        repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
        return repaired

    @classmethod
    def _parse_json(cls, raw: str) -> dict:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        span = cls._extract_json_span(cleaned)
        if span is None:
            raise ValueError(f"No balanced JSON object found in summarizer LLM response: {raw[:200]}")
        try:
            return json.loads(span)
        except json.JSONDecodeError as e:
            try:
                return json.loads(cls._repair_json(span))
            except json.JSONDecodeError:
                raise ValueError(
                    f"{e} | offending JSON span (len={len(span)}): {span}"
                ) from e

    def _summarize_one_batch(self, batch: list[dict], conference: str, year: str, _depth: int = 0) -> dict[int, str]:
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
            raw = self._call_llm(prompt, json_schema=SUMMARIES_JSON_SCHEMA)
            parsed = self._parse_json(raw)
            out = {}
            for entry in parsed.get("summaries", []):
                idx = entry.get("index")
                summary = (entry.get("summary") or "").strip()
                if idx is not None and summary:
                    out[int(idx)] = summary
            missing = [item["index"] for item in batch if item["index"] not in out]
            if missing:
                log.warning(f"Summarizer batch returned no entry for paper index(es) {missing} — those will fall back to raw excerpts")
            return out
        except Exception as e:  
            indices = [item["index"] for item in batch]
            if len(batch) > 1:
                log.warning(
                    f"Summarizer batch call failed for indices {indices} "
                    f"({type(e).__name__}: {e}) — bisecting and retrying the two halves separately"
                )
                mid = len(batch) // 2
                out = {}
                out.update(self._summarize_one_batch(batch[:mid], conference, year, _depth + 1))
                out.update(self._summarize_one_batch(batch[mid:], conference, year, _depth + 1))
                return out
            log.warning(
                f"Summarizer call failed for paper index {indices} even after bisecting to a single paper "
                f"({type(e).__name__}: {e}) — falling back to a raw abstract excerpt for this paper"
            )
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
        except Exception as e:  
            self.last_intro_error = f"{type(e).__name__}: {e}"
            log.warning(f"Lead-paragraph call failed ({self.last_intro_error}) — falling back to a templated intro")
            return ""



# Deterministic email assembly — no LLM involved past this point. 
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
        except Exception as e:  
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