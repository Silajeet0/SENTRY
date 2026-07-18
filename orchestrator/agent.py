"""
agent.py — the tool-calling loop on top of orchestrator.tools.

Uses the same OpenAI-compatible client + env-var configuration as
extractors/llm_extractor.py (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY,
LLM_BASE_URL — Groq's openai/gpt-oss-20b in the reference setup). Function
calling needs to actually be supported by whatever LLM_MODEL you point this
at; gpt-oss-20b via Groq supports it.

This is deliberately the only place in AEGIS where an LLM is allowed to plan
and take a sequence of actions. Everything the tools call into —
main_driver, pipeline.process_paper, the four scraper tiers, the single
per-paper extraction call — stays exactly as deterministic as it was before.
The agent only ever decides *which* deterministic tool to call and *in what
order*, based on what the person asks for.
"""
import json
import logging
import os

from orchestrator.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the AEGIS orchestrator — an assistant that runs the AEGIS \
academic paper extraction pipeline on behalf of a researcher.

IMPORTANT — what run_pipeline already does: identifying Indian-affiliated \
authors is not a separate filtering step you need to plan or ask about — \
it is the core, built-in function of AEGIS's per-paper extraction. Every \
paper run_pipeline processes is automatically scraped, checked for author \
affiliations, and classified as Indian-affiliated or not, with results \
written to disk. "Extract Indian-affiliated papers from <conference>" is \
fully satisfied by resolving the URL and calling run_pipeline — there is \
no extra filtering step to plan, ask about, or perform yourself.

You have tools to resolve conference proceedings URLs, validate them, detect \
their link structure, start extraction runs, check on run progress, and \
retry failed papers. You do not scrape or extract papers yourself — every \
tool call delegates to AEGIS's existing deterministic pipeline code. Never \
invent a proceedings URL yourself; always get it from resolve_conference_url \
or from the person.

Guidelines:
- Before starting a run for a conference, resolve its URL and validate it. \
If resolve_conference_url comes back resolved=false, ask the person for the \
URL instead of guessing one.
- resolve_conference_url returns mode="openreview_api" with a venue_id for \
OpenReview-hosted conferences (ICML, ICLR, and their oral/spotlight \
variants) instead of a proceeding_url. For those, call run_pipeline with \
venue_id (not proceeding_url) — that mode goes straight to OpenReview's \
API with no scraping or browser involved, and uses skip_venue_keywords / \
include_only_venue_keywords instead of skip_track_keywords / \
include_track_keywords. There's nothing to validate_url for venue_id mode \
— skip straight to run_pipeline once you have the venue_id.
- run_pipeline and retry_errors return immediately and run in the \
background — they do not block. After starting one or more runs, call \
get_run_status for each of them to report back what's actually happening; \
if a run is still early, say so honestly instead of implying it's done.
- When asked to "retry the errors" without a named conference, call \
list_runs first to see which runs actually have errors, then retry each \
of those — don't guess which one was meant.
- If the person says to skip or only include certain tracks (e.g. "skip \
workshop tracks"), pass that straight through as skip_track_keywords / \
include_track_keywords on run_pipeline — don't try to filter tracks \
yourself.
- Don't ask clarifying questions about things the tools already handle. \
If a request maps cleanly onto resolve_conference_url + run_pipeline, just \
do it.
- get_run_status may return "stale_data": true — this means the numbers on \
disk are leftover from a PREVIOUS run of that conference/year, not live \
progress of a run currently queued/extracting links. Say so plainly \
("still extracting links, no progress to report yet") rather than \
repeating the stale numbers as if they were current.
- get_run_status may report orchestrator_state "blocked" — this means the \
run detected repeated bot-challenge/rate-limit signals from the target \
site and stopped itself deliberately rather than continuing to hit a \
domain that's blocking this IP. Tell the person plainly that the site \
appears to be rate-limiting/blocking, that continuing to retry \
immediately is likely to make it worse, and suggest waiting a while \
(at least 30-60 minutes, longer for a harder block) before calling \
retry_errors — don't just retry immediately yourself.
- Never describe, suggest, or show a tool's name/arguments as something \
the person should run themselves — you have direct access to every tool \
listed here. If a tool call is the right next step, make it yourself; \
don't print its arguments as an example and tell the person to invoke it.
- Be concise. Report concrete numbers (papers found, errors, progress %) \
rather than vague status updates.

IKDD form-filler / RPA submission — a SEPARATE, downstream step from \
extraction: once run_pipeline has completed for a conference/year, the \
person may separately ask to "initiate the RPA process", "submit to IKDD", \
"upload the papers", or similar for that conference/year. That maps to \
initiate_form_filler, NOT run_pipeline — don't re-run extraction for a \
request like this, and don't start a form-filler run for a conference/year \
that hasn't been extracted yet (initiate_form_filler will fail cleanly if \
the extracted-papers file doesn't exist yet; if so, tell the person to run \
extraction first rather than retrying blindly).
- initiate_form_filler needs venue/month (the IKDD form's exact dropdown \
text) — if the person didn't give them, just call initiate_form_filler \
without them; it resolves them itself via resolve_ikdd_form_metadata and \
tells you plainly (status="needs_input") if it can't, at which point ask \
the person for the exact dropdown text. Never guess venue/month yourself.
- initiate_form_filler also runs in the background and returns immediately \
— poll get_rpa_status the same way you'd poll get_run_status, and report \
real submitted/skipped/failed counts once it finishes rather than assuming \
it's done right after queuing it.
- initiate_form_filler always re-checks IKDD's current New + Approved \
lists before submitting anything and skips papers already there — so it's \
safe to call again later (e.g. after a partial failure) without creating \
duplicate submissions.
- For a vague "check the upload status" or "retry the RPA" without a named \
conference, call list_rpa_runs first, the same way you'd call list_runs \
for a vague extraction-retry request.
"""


class Orchestrator:
    """
    One Orchestrator instance = one ongoing conversation. Create a new one
    per session; conversation history (including tool calls/results) lives
    in self.messages so multi-turn follow-ups like "retry the errors" have
    the full context of what was just run.
    """

    def __init__(self, max_tool_iterations: int = 12):
        self.provider = os.getenv("LLM_PROVIDER", "nvidia").lower()
        self.model = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.max_tool_iterations = max_tool_iterations
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # How long a single LLM call is allowed to run before giving up, and
        # how many times the SDK retries transient failures on its own
        # before that. Default SDK timeout is 600s (10 min) with no
        # visibility into *why* a call is slow — that reads as "hung" from
        # the UI either way, so some cap matters regardless of backend.
        # But the RIGHT cap differs by backend:
        #   - Remote/rate-limited (e.g. Groq): the orchestrator's own calls
        #     share the same API key/quota as the pipeline's own per-paper
        #     extraction calls, so a saturated rate limit is the likely
        #     slow-call cause — failing fast (short timeout) surfaces that
        #     quickly via the retry/error-message handling below.
        #   - Local (e.g. Ollama serving gpt-oss-20b on-device): there's no
        #     rate limit, but genuine compute contention is real if the
        #     pipeline's own extraction calls are hitting the same local
        #     model concurrently — a tool-schema-heavy prompt can
        #     legitimately take a while under load. A short timeout here
        #     would kill valid slow generations, not just stuck ones.
        # Overridable via env rather than hardcoded to either case.
        self.request_timeout_seconds = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "120"))
        self.request_max_retries = int(os.getenv("LLM_REQUEST_MAX_RETRIES", "1"))

        # Separate from the SDK-level retries above: a Groq 429 (TPM budget
        # exhausted, common on the on_demand free tier — see class docstring)
        # is NOT the same failure as the model hallucinating an invalid tool
        # call, and shouldn't be handled the same way. Retrying with
        # tool_choice="none" (the fallback further down, meant for that
        # other case) does nothing to fix a token-budget deficit, and can
        # itself 400 if the model tries to call a tool anyway despite
        # tool_choice="none" — seen in practice with gpt-oss-20b. So a 429
        # gets its own backoff-and-retry loop with tool_choice="auto"
        # unchanged, waiting for the budget to actually refill.
        self.rate_limit_max_retries = int(os.getenv("LLM_RATE_LIMIT_MAX_RETRIES", "3"))
        self.rate_limit_base_delay_seconds = float(os.getenv("LLM_RATE_LIMIT_BASE_DELAY_SECONDS", "20"))

    def _client(self):
        from openai import OpenAI
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_seconds,
            max_retries=self.request_max_retries,
        )

    def _rate_limit_delay_seconds(self, error, attempt: int) -> float:
        """
        Groq's 429 body usually includes a concrete "Please try again in
        19.5525s" (or "...885ms") hint — honor that plus a small buffer
        when present, since it's the actual TPM-window reset time. Fall
        back to a growing fixed delay otherwise.
        """
        import re
        match = re.search(r"try again in ([\d.]+)\s*(ms|s)\b", str(error))
        if match:
            value, unit = match.groups()
            seconds = float(value) / 1000.0 if unit == "ms" else float(value)
            return seconds + 1.0
        return self.rate_limit_base_delay_seconds * (attempt + 1)

    def _call_llm(self, client):
        """
        One resilient LLM call, covering the two failure modes seen in
        practice with Groq's gpt-oss-20b separately rather than through one
        shared fallback:
          - RateLimitError (429): back off for the API's own suggested
            window and retry with tool_choice UNCHANGED (still "auto") —
            the problem is token budget, not tool-calling behavior.
          - Anything else (e.g. a hallucinated/invalid tool call name,
            which Groq rejects server-side before we see a message): retry
            once with tool_choice="none" as before.
        Raises the last error if every retry is exhausted.
        """
        from openai import RateLimitError
        import time

        last_error = None
        for attempt in range(self.rate_limit_max_retries + 1):
            try:
                return client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except RateLimitError as e:
                last_error = e
                if attempt >= self.rate_limit_max_retries:
                    break
                delay = self._rate_limit_delay_seconds(e, attempt)
                log.warning(
                    f"Rate limited (attempt {attempt + 1}/{self.rate_limit_max_retries + 1}) "
                    f"— waiting {delay:.1f}s for the TPM budget to refill: {e}"
                )
                time.sleep(delay)
            except Exception as e:  # noqa: BLE001
                log.warning(f"LLM call failed ({type(e).__name__}: {e}) — retrying with tool_choice='none'")
                try:
                    return client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice="none",
                        temperature=0.2,
                    )
                except Exception as e2:  # noqa: BLE001
                    last_error = e2
                    break

        raise last_error

    def chat(self, user_message: str, on_tool_call=None) -> str:
        """
        Send a user message, run the tool-calling loop until the model
        produces a final text reply (no more tool calls), and return that
        reply.

        on_tool_call, if given, is called as on_tool_call(name, args, result)
        immediately after each tool executes — useful for printing a live
        trace in a CLI or UI.
        """
        self.messages.append({"role": "user", "content": user_message})
        client = self._client()

        for iteration in range(self.max_tool_iterations):
            try:
                response = self._call_llm(client)
            except Exception as e:  # noqa: BLE001
                log.exception("LLM call failed after all retries")
                return (
                    "I hit an error talking to the model while wrapping "
                    "up this turn. Any tool calls made earlier in this "
                    "turn (starting a run, checking status, etc.) may "
                    "have already succeeded regardless — check "
                    f"get_run_status if that's a concern. Underlying "
                    f"error: {type(e).__name__}: {e}"
                )

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)

            assistant_msg = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_msg)

            if not tool_calls:
                return message.content or ""

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                func = TOOL_FUNCTIONS.get(name)
                if func is None:
                    result = {"error": f"Unknown tool '{name}'"}
                else:
                    try:
                        result = func(**args)
                    except Exception as e:  # noqa: BLE001 — must surface to the model, not crash the loop
                        log.exception(f"Tool {name} raised")
                        result = {"error": f"{type(e).__name__}: {e}"}

                if on_tool_call:
                    on_tool_call(name, args, result)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    }
                )

        return (
            "I hit the tool-call limit for this turn without wrapping up — "
            "ask me to continue and I'll pick back up from here."
        )
