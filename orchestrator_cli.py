"""
orchestrator_cli.py — interactive REPL for the AEGIS agentic orchestrator.

    python orchestrator_cli.py

Uses the same .env / LLM_* environment variables as the rest of AEGIS
(see README.md — LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL).

Example session:

    you> Extract Indian-affiliated papers from NeurIPS 2025, ICML 2025, and
         ACL 2025. Skip workshop tracks.
    ⚙️  resolve_conference_url(conference='NeurIPS', year='2025')
         → {'conference': 'NeurIPS', 'year': '2025', 'proceeding_url': ...
    ...
    agent> Started all three runs in the background — NeurIPS and ACL are
    flat/grouped respectively, ICML is going through OpenReview. I'll check
    back on progress if you ask for a status update.

    you> what's the status?
    ...

    you> retry the errors
    ...

Ctrl-C or "exit" to quit.
"""
import logging

from dotenv import load_dotenv

load_dotenv()

from orchestrator.agent import Orchestrator  # noqa: E402  (after load_dotenv on purpose)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _print_tool_call(name: str, args: dict, result) -> None:
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"  ⚙️  {name}({args_str})")
    preview = str(result)
    if len(preview) > 300:
        preview = preview[:300] + "…"
    print(f"     → {preview}")


def main() -> None:
    print("AEGIS orchestrator — type an instruction, or 'exit' to quit.\n")
    orch = Orchestrator()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        reply = orch.chat(user_input, on_tool_call=_print_tool_call)
        print(f"\nagent> {reply}\n")


if __name__ == "__main__":
    main()
