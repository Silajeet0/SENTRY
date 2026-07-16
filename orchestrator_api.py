"""
orchestrator_api.py — minimal OpenAI-compatible /v1/chat/completions server
wrapping the AEGIS orchestrator, so a self-hosted OpenWebUI can talk to it
as a custom "Connection".

    pip install fastapi uvicorn --break-system-packages
    python orchestrator_api.py            # listens on 0.0.0.0:8091

Then in OpenWebUI: Settings > Admin Settings > Connections > OpenAI API >
add a connection with Base URL "http://host.docker.internal:8091/v1"
(or your host's LAN IP if OpenWebUI is on another machine) and any
placeholder API key — this server doesn't check it. A model called
"aegis-orchestrator" will then show up in the model picker.

Privacy note: this server and OpenWebUI are both fully self-hosted and
neither phones home. The only outbound network call in the whole stack is
the orchestrator's own LLM call — to whatever LLM_BASE_URL is set to in
your .env (Groq by default). If you want zero external traffic at all,
point LLM_BASE_URL at a local Ollama instance instead (see README's
Agentic Orchestrator section) — everything else here works unchanged.

Design note: this endpoint is intentionally stateless per HTTP request.
OpenWebUI resends the full visible conversation with every call, like any
OpenAI chat client, so each request rebuilds an Orchestrator from that
history rather than tracking sessions server-side. Multi-turn things like
"retry the errors" still work correctly because actual AEGIS run state
lives in the process-wide orchestrator.registry.REGISTRY — the agent just
calls list_runs()/get_run_status() again each turn to see current reality,
regardless of whether the Python-side conversation object persisted.
"""
import time
import uuid
from typing import List

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from orchestrator.agent import Orchestrator  # noqa: E402  (after load_dotenv on purpose)

app = FastAPI(title="AEGIS Orchestrator (OpenAI-compatible)")


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "aegis-orchestrator"
    messages: List[ChatMessage]
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "aegis-orchestrator", "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
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
        except Exception as e:  # noqa: BLE001
            # Belt-and-suspenders on top of agent.py's own retry/recovery —
            # OpenWebUI shows this as the assistant's message rather than a
            # raw connection error, and any tool calls already made this
            # turn (e.g. a run that already started) aren't affected either
            # way since they execute independently in background threads.
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
