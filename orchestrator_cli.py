"""
orchestrator_cli.py — interactive REPL for the SENTRY agentic orchestrator.

    python orchestrator_cli.py

Uses the same .env / LLM_* environment variables as the rest of SENTRY
(see README.md — LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL).

"""
import logging

from dotenv import load_dotenv

load_dotenv()

from orchestrator.agent import Orchestrator 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _print_tool_call(name: str, args: dict, result) -> None:
    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"[TOOL] {name}({args_str})")
    preview = str(result)
    if len(preview) > 300:
        preview = preview[:300] + "…"
    print(f"     → {preview}")


def main() -> None:
    print("SENTRY orchestrator — type an instruction, or 'exit' to quit.\n")
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
