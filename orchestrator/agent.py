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

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

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
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except Exception as e:  # noqa: BLE001
                # gpt-oss-20b via Groq occasionally hallucinates a tool call
                # to a name that isn't in TOOL_SCHEMAS (seen in practice:
                # inventing a "json" tool to wrap its own summary instead of
                # just replying in plain text). Groq validates tool calls
                # server-side and rejects it with 400 before we ever see a
                # message object, so this raises here — not in the
                # tool-dispatch try/except below, and not something that
                # means any real tool calls made earlier this turn failed.
                log.warning(f"LLM call failed ({type(e).__name__}: {e}) — retrying with tool_choice='none'")
                try:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        tools=TOOL_SCHEMAS,
                        tool_choice="none",
                        temperature=0.2,
                    )
                except Exception as e2:  # noqa: BLE001
                    log.exception("LLM call failed on retry too")
                    return (
                        "I hit an error talking to the model while wrapping "
                        "up this turn. Any tool calls made earlier in this "
                        "turn (starting a run, checking status, etc.) may "
                        "have already succeeded regardless — check "
                        f"get_run_status if that's a concern. Underlying "
                        f"error: {type(e2).__name__}: {e2}"
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
