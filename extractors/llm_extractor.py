"""
llm_extractor
"""
import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional
from evaluation.laya_classifier import classify_affiliation

AREA_LIST = [
    "AI", "Database", "Machine Learning", "NLP", "Speech", "Vision",
    "Extreme Classification", "Game Theory and Economic Paradigms",
    "Graph Analytics", "Image captioning", "Information Retrieval/Ranking",
    "Machine Learning for Computer Graphics", "Open Domain Generalization", "Security",
    "Systems and Infrastructure for Web, Mobile Web, and Web of Things", "System Reliability", "Others"
]

#to prevent non-determinisim in the RPA
AREA_ALIASES = {
    "deep learning": "Machine Learning",
    "neural networks": "Machine Learning",
    "representation learning": "Machine Learning",
    "self-supervised learning": "Machine Learning",
    "reinforcement learning": "Machine Learning",
    "large language models": "NLP",
    "llms": "NLP",
    "language models": "NLP",
    "generative ai": "AI",
    "computer vision": "Vision",
    "object detection": "Vision",
    "image generation": "Vision",
    "diffusion models": "Vision",
    "speech recognition": "Speech",
    "speech synthesis": "Speech",
    "search": "Information Retrieval/Ranking",
    "retrieval": "Information Retrieval/Ranking",
}


@dataclass
class PaperInfo:
    paper_url: str
    paper_title: str = ""
    area_of_research: str = ""
    # Populated ONLY when area_of_research falls back to "Others" and the
    # LLM's own free-text answer had something more specific to say than
    # that (e.g. "Federated Learning") — the IKDD (or any other suitable venue) form reveals a follow-up
    # "Enter the area of research" text box the moment "Others" is picked
    # on the dropdown, and Form_filler needs this to fill it in
    area_of_research_other: str = ""
    total_authors: int = 0
    all_authors: list = field(default_factory=list)
    authors_with_indian_affiliations: list = field(default_factory=list)
    indian_institutions: list = field(default_factory=list)
    source: str = ""
    raw_content_source: str = ""
    error: str = ""
    abstract: str = ""


EXTRACTION_PROMPT = """You are a precise data extractor for academic papers. 
Extract the requested information from the page content below and return ONLY valid JSON.

AREAS OF RESEARCH (pick exactly one):
{area_list}

If the paper's topic does not clearly fit any option above other than "Others", set
"area_of_research" to "Others" AND set "area_of_research_other" to a short (2-6 word)
specific description of the paper's actual research area, e.g. "Algebraic Complexity
Theory", "Cryptographic Protocols", "Federated Learning". If "area_of_research" is
anything OTHER than "Others", leave "area_of_research_other" as an empty string "".

AFFILIATION EXTRACTION:
- Extract the official institutional affiliation string associated with each author.
- Preserve location, campus, department, company, and country text exactly enough for a downstream classifier to evaluate it.
- Ignore author names and email addresses as evidence; they may be retained only when they are part of the source's author record.
- Do not classify affiliations or infer whether they are Indian. The downstream fine-tuned Laya classifier makes that decision.

PAGE CONTENT:
{content}

Return ONLY this JSON (no markdown, no explanation):
{{
  "paper_title": "exact title as it appears",
  "abstract": "abstract if present, otherwise empty string",
  "area_of_research": "one area from the list above",
  "area_of_research_other": "short specific area description, ONLY if area_of_research is 'Others', otherwise empty string",
  "total_authors": 0,
  "all_authors": [
    {{"name": "Full Name", "affiliation": "Full affiliation string or Unknown"}}
  ]
}}

"""

# JSON schema mirroring the extraction shape. It is sent through each provider's native constrained-
# decoding interface, so the paper-metadata and affiliation extraction are
# generated as schema-valid JSON rather than merely being asked for JSON in
# the prompt.
EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "paper_title": {"type": "string"},
        "abstract": {"type": "string"},
        "area_of_research": {"type": "string", "enum": AREA_LIST},
        "area_of_research_other": {"type": "string"},
        "total_authors": {"type": "integer"},
        "all_authors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "affiliation": {"type": "string"},
                },
                "required": ["name", "affiliation"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "paper_title",
        "abstract",
        "area_of_research",
        "area_of_research_other",
        "total_authors",
        "all_authors",
    ],
    "additionalProperties": False,
}


class LLMExtractor:

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "nvidia").lower()
        self.model = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.base_url = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    def extract(self, content: str, url: str, content_source: str) -> PaperInfo:
        """
        Single LLM call to extract all paper info from raw content.
        Returns PaperInfo with error field set on failure.
        """
        info = PaperInfo(paper_url=url, raw_content_source=content_source)

        try:
            prompt = EXTRACTION_PROMPT.format(
                area_list="\n".join(f"- {a}" for a in AREA_LIST),
                content=content[:12000]   # hard cap so we don't face any issue with context limits
            )

            parsed = None
            last_error = None
            raw = ""
            for attempt in range(2):
                retry_instruction = (
                    "\n\nIMPORTANT: Your previous response was not valid JSON. "
                    "Return one valid JSON object only, with double-quoted keys and strings."
                )
                raw = self._call_llm(
                    prompt + (retry_instruction if attempt else ""),
                    json_schema=EXTRACTION_JSON_SCHEMA,
                )
                try:
                    parsed = self._parse_json(raw)
                    break
                except Exception as e:
                    last_error = e

            if parsed is None:
                raise last_error or ValueError(f"No valid JSON found in LLM response: {raw[:200]}")

            info.paper_title = parsed.get("paper_title", "")
            info.abstract = parsed.get("abstract", "")
            normalized_area, derived_other = self._normalize_area_of_research(
                parsed.get("area_of_research", "Others")
            )
            info.area_of_research = normalized_area
            info.area_of_research_other = self._resolve_other_text(
                parsed.get("area_of_research_other", ""), derived_other, normalized_area
            )
            info.all_authors = self._normalize_authors(parsed.get("all_authors", []))
            info.total_authors = len(info.all_authors)

            # Classification is intentionally a separate, validated Laya call.
            # The extraction model never decides geography; the lists below
            # are populated only from validated Laya decisions.
            info = self._classify_authors_with_laya(info)

        except Exception as e:
            info.error = str(e)

        return info

    def _call_llm(self, prompt: str, json_schema: Optional[dict] = None) -> str:
        """Call the LLM, using provider-native constrained JSON decoding."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        request = dict(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract structured paper metadata. Return only a valid JSON object."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # we don't need creativity
            max_tokens=2000,
        )
        extra_body = {}
        constrained = bool(json_schema and self._should_request_json_mode())
        if constrained and self.provider in {"openai", "groq"}:
            # OpenAI-style Structured Outputs. strict=True is token-level
            # constrained decoding on supported models (including the Groq
            # GPT-OSS models used by this project).
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "paper_affiliation_extraction",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        elif constrained and self.provider == "nvidia":
            # NVIDIA NIM exposes JSON-schema constrained decoding through its
            # OpenAI-compatible nvext extension.
            extra_body["nvext"] = {"guided_json": json_schema}
        elif constrained and (
            self.provider == "ollama"
            or any(host in self.base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))
        ):
            # Ollama's native `format` field compiles the schema to a grammar.
            extra_body["format"] = json_schema
        elif self._should_request_json_mode():
            # Unknown OpenAI-compatible providers may only implement JSON
            # object mode. This is intentionally not used for the known
            # providers above because it does not enforce the schema.
            request["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**request, extra_body=extra_body or None)

        message = response.choices[0].message
        content = message.content or ""
        return content.strip()

    def _should_request_json_mode(self) -> bool:
        value = os.getenv("LLM_RESPONSE_FORMAT_JSON", "auto").lower()
        if value in {"1", "true", "yes"}:
            return True
        if value in {"0", "false", "no"}:
            return False
        return (
            "groq.com" in self.base_url
            or self.provider in {"openai", "groq", "ollama", "nvidia"}
            or any(host in self.base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))
        )

    def _parse_json(self, raw: str) -> dict:
        """Extract JSON from LLM response robustly."""
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        # Find the outermost JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"No valid JSON found in LLM response: {raw[:200]}")

    _EMPTY_OTHER_VALUES = {"other", "others", "n/a", "na", "none", "unknown", ""}

    def _normalize_area_of_research(self, area: object) -> tuple[str, str]:
        """
        Return (area_of_research, area_of_research_other) — an IKDD (or any other suitable venue)
        form-safe dropdown value from the LLM's free text, plus (when the
        dropdown value is "Others") the LLM's original wording so it can
        go into the form's follow-up "Enter the area of research" text box
        instead of being thrown away.
        """
        if not isinstance(area, str):
            return "Others", ""

        cleaned = re.sub(r"\s+", " ", area).strip()
        valid_by_lower = {valid.lower(): valid for valid in AREA_LIST}
        lowered = cleaned.lower()

        if lowered in valid_by_lower:
            return valid_by_lower[lowered], ""

        if lowered in AREA_ALIASES:
            return AREA_ALIASES[lowered], ""

        if lowered in self._EMPTY_OTHER_VALUES:
            return "Others", ""

        return "Others", cleaned

    def _resolve_other_text(self, explicit_other: object, derived_other: str, area: str) -> str:
        """
        Pick the text to show in the IKDD (or any other suitable venue) form's follow-up "Enter the area
        of research" box, when area_of_research is "Others". 

        Returns "" if area isn't "Others" at all, or if neither source has
        anything usable.
        """
        if area != "Others":
            return ""

        if isinstance(explicit_other, str):
            cleaned = re.sub(r"\s+", " ", explicit_other).strip()
            if cleaned and cleaned.lower() not in self._EMPTY_OTHER_VALUES:
                return cleaned

        return derived_other

    def _normalize_authors(self, authors: object) -> list[dict[str, str]]:
        """Normalize the extractor's author records before they reach Laya."""
        if not isinstance(authors, list):
            return []
        normalized = []
        for author in authors:
            if not isinstance(author, dict):
                continue
            name = author.get("name", "")
            affiliation = author.get("affiliation", "")
            if not isinstance(name, str) or not isinstance(affiliation, str):
                continue
            name = re.sub(r"\s+", " ", name).strip()
            affiliation = re.sub(r"\s+", " ", affiliation).strip()
            if name:
                normalized.append({"name": name, "affiliation": affiliation or "Unknown"})
        return normalized

    def _classify_authors_with_laya(self, info: PaperInfo) -> PaperInfo:
        """Populate the legacy output lists from Laya's binary decisions."""
        indian_authors = []
        indian_institutions = []
        for author in info.all_authors:
            affiliation = author.get("affiliation", "")
            if not affiliation or affiliation.strip().lower() == "unknown":
                continue
            decision = classify_affiliation(affiliation)
            if decision.label == "INDIAN":
                indian_authors.append(author["name"])
                indian_institutions.append(affiliation)
        info.authors_with_indian_affiliations = sorted(set(indian_authors))
        info.indian_institutions = sorted(set(indian_institutions))
        return info
