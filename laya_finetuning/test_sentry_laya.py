#!/usr/bin/env python3
"""Smoke-test a locally copied, fine-tuned SENTRY-Laya checkpoint.

Examples:
    python test_sentry_laya.py --model ./laya_finetuned_SENTRY
    python test_sentry_laya.py --model ./laya_finetuned_SENTRY --device cpu
    python test_sentry_laya.py --model ./laya_finetuned_SENTRY --text "IIT Bombay, Mumbai"
"""

import argparse
from pathlib import Path
import sys

import torch
import laya

# Keep identical to the questions/criteria used to fine-tune the model.
QUESTIONS = {
    "affiliation_geography": {
        "type": "choice",
        "instructions": (
            "Does this affiliation provide sufficient evidence that the current "
            "institution or campus is located in India?"
        ),
        "criteria": {
            "INDIAN": "Sufficient evidence that the current institution or campus is in India.",
            "NOT_INDIAN": "No sufficient evidence that the current institution or campus is in India.",
        },
    }
}

EXAMPLES = [
    "Indian Institute of Technology Bombay, Mumbai, India",
    "Microsoft Research India, Bengaluru",
    "University of Cambridge, United Kingdom",
    "Microsoft Research",
    "Indian Institute of Technology Madras, Zanzibar, Tanzania",
    "researcher@iitb.ac.in",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="./laya_finetuned_SENTRY", help="Local exported checkpoint directory")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--text", help="Classify one affiliation instead of the built-in examples")
    args = parser.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    if not model_dir.is_dir():
        parser.error(f"Model directory not found: {model_dir}")
    if not (model_dir / "rl_agent_config.json").is_file():
        parser.error(f"Missing rl_agent_config.json in {model_dir}; check your transferred checkpoint")
    if not list(model_dir.glob("*.safetensors")):
        print("Warning: no *.safetensors found at checkpoint root; inspect the exported files.", file=sys.stderr)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA selected but not available. Try --device cpu or install CUDA-enabled PyTorch.")

    print(f"PyTorch: {torch.__version__}")
    print(f"Checkpoint: {model_dir}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading Laya checkpoint...")
    agent = laya.load(str(model_dir), device=device)
    print("Model loaded successfully.\n")

    tests = [args.text] if args.text is not None else EXAMPLES
    for i, text in enumerate(tests, 1):
        output = agent.predict(text, QUESTIONS)
        answer = output["answers"]["affiliation_geography"]
        label = answer["choice"]
        probabilities = answer.get("probabilities", {})
        print(f"[{i}] Affiliation: {text}")
        print(f"    Prediction: {label}")
        if isinstance(probabilities, dict):
            for cls in ("INDIAN", "NOT_INDIAN"):
                if cls in probabilities:
                    print(f"    P({cls}): {float(probabilities[cls]):.4f}")
        print()

    print("PASS: model loading and inference completed.")
    print("Note: successful inference does not establish classification accuracy.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
