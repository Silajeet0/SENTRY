"""
orchestrator_api.py — minimal OpenAI-compatible /v1/chat/completions server
wrapping the SENTRY orchestrator, so a self-hosted OpenWebUI can talk to it
as a custom "Connection".

    pip install fastapi uvicorn --break-system-packages
    python orchestrator_api.py            # listens on 0.0.0.0:8091

Then in OpenWebUI: Settings > Admin Settings > Connections > OpenAI API >
add a connection with Base URL "http://host.docker.internal:8091/v1"
(or your host's LAN IP if OpenWebUI is on another machine) and any
placeholder API key — this server doesn't check it. A model called
"sentry-orchestrator" will then show up in the model picker.

"""
import time
import uuid
from typing import List

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI  
from pydantic import BaseModel  

from orchestrator.agent import Orchestrator 

app = FastAPI(title="SENTRY Orchestrator (OpenAI-compatible)")


class ChatMessage(BaseModel):
    role: str
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "sentry-orchestrator"
    messages: List[ChatMessage]
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "sentry-orchestrator", "object": "model", "owned_by": "local"}],
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
