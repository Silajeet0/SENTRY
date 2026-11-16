"""Production affiliation classification with the fine-tuned SENTRY Laya model.

The extractor is responsible for finding the official affiliation string.  This
module is the only production component that decides whether that string is
evidence of an institution or campus in India.  It deliberately has no rule
fallback: a missing or invalid model response is an error rather than an
implicit deterministic classification.
"""
from __future__ import annotations

import importlib.metadata
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


QUESTION_ID = "affiliation_geography"
LABELS = ("INDIAN", "NOT_INDIAN")
DEFAULT_LAYA_THRESHOLD = 0.9781
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "laya_finetuning" / "laya_finetuned_SENTRY"

QUESTIONS = {
    QUESTION_ID: {
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


def build_laya_request(affiliation: str) -> dict[str, Any]:
    """Build the inference payload shape used by the fine-tuning corpus."""
    if not isinstance(affiliation, str) or not affiliation.strip():
        raise ValueError("Affiliation must be a non-empty string")
    return {"state": affiliation.strip(), "questions": QUESTIONS}


@dataclass(frozen=True)
class LayaDecision:
    """Validated result returned by the production checkpoint."""

    label: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    response: dict[str, Any] = field(default_factory=dict, repr=False)
    raw_label: str | None = None
    max_probability: float | None = None
    indian_probability: float | None = None
    threshold: float = DEFAULT_LAYA_THRESHOLD

    @property
    def matches(self) -> list[str]:
        """Compatibility field for old report consumers; Laya has no rules."""
        return []


class LayaAffiliationClassifier:
    """Lazy, reusable wrapper around one fine-tuned Laya agent."""

    def __init__(
        self,
        model: str | Path | None = None,
        device: str | None = None,
        threads: int | None = None,
        threshold: float = DEFAULT_LAYA_THRESHOLD,
    ):
        configured_model = model or os.getenv("SENTRY_LAYA_MODEL")
        self._requires_local_checkpoint = configured_model is None
        self.model = str(configured_model or DEFAULT_MODEL)
        self.device = device or os.getenv("SENTRY_LAYA_DEVICE", "cpu")
        self.threads = int(threads or os.getenv("SENTRY_LAYA_THREADS", "2"))
        if self.threads < 1:
            raise ValueError("SENTRY_LAYA_THREADS must be at least 1")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("SENTRY_LAYA_THRESHOLD must be a number in [0, 1]")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("SENTRY_LAYA_THRESHOLD must be a number in [0, 1]")
        self.threshold = float(threshold)
        self._agent = None
        self._load_lock = threading.Lock()

    def _load(self):
        if self._agent is not None:
            return self._agent
        with self._load_lock:
            if self._agent is not None:
                return self._agent
            os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "data" / "model_cache"))
            os.environ.setdefault("USE_TF", "0")
            import torch
            import laya

            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("SENTRY_LAYA_DEVICE=cuda but CUDA is not available")
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.set_num_threads(self.threads)
            torch.manual_seed(0)
            location = Path(self.model).expanduser()
            if self._requires_local_checkpoint and not location.is_dir():
                raise FileNotFoundError(
                    f"Fine-tuned SENTRY Laya checkpoint not found at {location}. "
                    "Set SENTRY_LAYA_MODEL to a local checkpoint directory or an explicit model ID."
                )
            if location.exists() and not location.is_dir():
                raise ValueError(f"SENTRY_LAYA_MODEL must be a checkpoint directory: {location}")
            self._agent = laya.load(str(location) if location.exists() else self.model, device=self.device)
        return self._agent

    def predict(self, affiliation: str) -> LayaDecision:
        request = build_laya_request(affiliation)
        response = self._load().predict("Current affiliation: " + request["state"], request["questions"])
        try:
            answer = response["answers"][QUESTION_ID]
            choice = answer["choice"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Laya response has no {QUESTION_ID} choice: {response!r}") from exc
        if choice not in LABELS:
            raise ValueError(f"Fine-tuned Laya returned unsupported label {choice!r}")

        probabilities = answer.get("probabilities") or {}
        if set(probabilities) != set(LABELS):
            raise ValueError(
                f"Laya probabilities do not contain exactly {LABELS}; "
                "probabilities are required for thresholded classification: "
                f"{probabilities!r}"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in probabilities.values()
        ):
            raise ValueError(f"Laya returned invalid probabilities: {probabilities!r}")
        total = sum(probabilities.values())
        if total <= 0 or not math.isclose(total, 1.0, abs_tol=0.002):
            raise ValueError(f"Laya probabilities do not sum to one: {probabilities!r}")
        normalized = {label: float(probabilities[label]) / total for label in LABELS}

        # Retain the raw argmax for diagnostics, but apply the calibrated
        # threshold to P(INDIAN), as selected on the development set.  The
        # binary policy has no abstention class, so a below-threshold result is
        # conservatively treated as NOT_INDIAN.
        raw_label = max(LABELS, key=normalized.__getitem__)
        max_probability = normalized[raw_label]
        indian_probability = normalized["INDIAN"]
        label = "INDIAN" if indian_probability >= self.threshold else "NOT_INDIAN"

        confidence = answer.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError(f"Laya returned invalid confidence: {confidence!r}")
        return LayaDecision(
            label=label,
            probabilities=normalized,
            confidence=float(confidence) if confidence is not None else None,
            response=response,
            raw_label=raw_label,
            max_probability=max_probability,
            indian_probability=indian_probability,
            threshold=self.threshold,
        )


_CLASSIFIER: LayaAffiliationClassifier | None = None
_CLASSIFIER_LOCK = threading.Lock()


def get_classifier() -> LayaAffiliationClassifier:
    """Return the process-wide lazy classifier used by the pipeline."""
    global _CLASSIFIER
    if _CLASSIFIER is None:
        with _CLASSIFIER_LOCK:
            if _CLASSIFIER is None:
                _CLASSIFIER = LayaAffiliationClassifier()
    return _CLASSIFIER


def classify_affiliation(affiliation: str) -> LayaDecision:
    """Classify one official affiliation with fine-tuned Laya."""
    return get_classifier().predict(affiliation)


def classify_affiliations(affiliations: list[str]) -> list[LayaDecision]:
    return [classify_affiliation(affiliation) for affiliation in affiliations]


def combine_author_decisions(decisions: list[LayaDecision]) -> str:
    """Apply SENTRY's positive-only inclusion policy to binary Laya labels."""
    return "INDIAN" if any(decision.label == "INDIAN" for decision in decisions) else "NOT_INDIAN"


def runtime_metadata() -> dict[str, Any]:
    """Return configuration metadata without loading the model."""
    versions = {}
    for package in ("laya", "torch", "transformers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    classifier = get_classifier()
    return {
        "model": classifier.model,
        "device": classifier.device,
        "threads": classifier.threads,
        "threshold": classifier.threshold,
        "versions": versions,
    }
