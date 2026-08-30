"""
orchestrator_api.py — minimal OpenAI-compatible /v1/chat/completions server
wrapping the SENTRY orchestrator, so a self-hosted OpenWebUI (or the sentry
pip client) can talk to it as a custom "Connection".

    pip install fastapi uvicorn --break-system-packages
    python orchestrator_api.py            # listens on 0.0.0.0:8091

Auth: set SENTRY_API_KEYS in .env as a comma-separated list of accepted
keys, e.g. SENTRY_API_KEYS=key-for-alice,key-for-bob. Requests must send
Authorization: Bearer <key>. If SENTRY_API_KEYS is unset, auth is skipped
entirely (local/dev use, e.g. plain OpenWebUI on localhost) — set it before
exposing this over the Cloudflare Tunnel.

Then in OpenWebUI: Settings > Admin Settings > Connections > OpenAI API >
add a connection with Base URL "http://host.docker.internal:8091/v1"
(or your host's LAN IP if OpenWebUI is on another machine) and one of the
SENTRY_API_KEYS values as the API key. A model called "sentry-orchestrator"
will then show up in the model picker.

"""
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from orchestrator.agent import Orchestrator

app = FastAPI(title="SENTRY Orchestrator (OpenAI-compatible)")

_API_KEYS = {k.strip() for k in os.getenv("SENTRY_API_KEYS", "").split(",") if k.strip()}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# Resolved once at process start — this is exactly the value that changes
# when the watcher (scripts/watch_and_restart.sh) restarts the service on a
# new commit, so /v1/version is also the cheapest way to confirm a restart
# actually picked up the latest code.
_SERVER_COMMIT = _git_commit()


def require_api_key(authorization: Optional[str] = Header(None)) -> None:
    if not _API_KEYS:
        return  # auth disabled — no SENTRY_API_KEYS configured
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token not in _API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "sentry-orchestrator"
    messages: List[ChatMessage]
    stream: bool = False


@app.get("/v1/version")
def version():
    # Unauthenticated on purpose: cheap liveness/staleness check for the
    # watcher and for teammates without needing a key handy.
    return {"commit": _SERVER_COMMIT, "auth_enabled": bool(_API_KEYS)}


@app.get("/v1/summary/{conference}/{year}")
def get_summary_content(conference: str, year: str, _: None = Depends(require_api_key)):
    """
    Serves an already-generated email_summary.json straight off disk — no
    orchestrator/LLM call involved. This exists specifically because asking
    the orchestrator to "show me the summary" makes the 20B model re-compose
    the entire cached digest as fresh output inside one synchronous request,
    which for a large digest can exceed Cloudflare's ~100-120s proxy read
    timeout (a 524, not adjustable on free tiers) — and worse, the
    generation keeps running on the Mac's GPU even after the client sees
    that timeout, since a disconnected client doesn't cancel an in-flight
    sync thread. A plain file read can't hit either problem.
    """
    path = Path(f"data/final_output/{conference}/{year}/email_summary.json")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No email_summary.json for {conference} {year} — run summarize_indian_authors first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/v1/tools/summarize/{conference}/{year}")
def trigger_summary(conference: str, year: str, refresh_cache: bool = False, _: None = Depends(require_api_key)):
    """
    Calls orchestrator.tools.summarize_indian_authors directly — no
    orchestrator LLM call at all, so this can't be slowed down by a cold
    Ollama model load or multi-step tool-selection reasoning the way asking
    the orchestrator to "summarize X" in chat can be. summarize_indian_authors
    itself only checks disk and enqueues onto summary_runner's background
    worker thread (milliseconds), so this endpoint is always fast regardless
    of whether the actual job then takes minutes — poll /v1/summary or
    get_summary_status for progress.
    """
    from orchestrator.tools import summarize_indian_authors

    return summarize_indian_authors(conference=conference, year=year, refresh_cache=refresh_cache)


@app.get("/v1/models")
def list_models(_: None = Depends(require_api_key)):
    return {
        "object": "list",
        "data": [{"id": "sentry-orchestrator", "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, _: None = Depends(require_api_key)):
    orch = Orchestrator()

    # Only real conversation turns matter — drop any system message
    # OpenWebUI might inject; Orchestrator already carries its own.
    history = [m for m in req.messages if m.role in ("user", "assistant") and m.content]

    if not history:
        reply = "Say something and I'll get started."
    else:
        *prior_turns, last_turn = history
        orch.messages += [{"role": m.role, "content": m.content} for m in prior_turns]
        try:
            reply = orch.chat(last_turn.content)
        except Exception as e:
            reply = (
                "Something went wrong finishing this response. Any run or "
                "status check requested this turn may have already gone "
                f"through regardless. Error: {type(e).__name__}: {e}"
            )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8091)
