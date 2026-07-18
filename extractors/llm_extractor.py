"""
In contrast to Agent-E, the LLM does ONLY extraction — never navigation or planning.
"""
import os
import json
import re
from dataclasses import dataclass, field
from typing import Optional
from evaluation.india_rules import classify_affiliation

AREA_LIST = [
    "AI", "Database", "Machine Learning", "Deep Learning", "NLP", "Speech", "Vision",
    "Extreme Classification", "Game Theory and Economic Paradigms",
    "Graph Analytics", "Image captioning", "Information Retrieval/Ranking",
    "Machine Learning for Computer Graphics", "Security", "System Reliability",
    "Systems and Infrastructure for Web, Mobile Web, and Web of Things", "Others"
]


@dataclass
class PaperInfo:
    paper_url: str
    paper_title: str = ""
    area_of_research: str = ""
    total_authors: int = 0
    all_authors: list = field(default_factory=list)
    authors_with_indian_affiliations: list = field(default_factory=list)
    indian_institutions: list = field(default_factory=list)
    source: str = ""
    raw_content_source: str = ""
    error: str = ""


EXTRACTION_PROMPT = """You are a precise data extractor for academic papers. 
Extract the requested information from the page content below and return ONLY valid JSON.

AREAS OF RESEARCH (pick exactly one):
{area_list}

INDIAN AFFILIATION INDICATORS & RULES:
- "India" mentioned explicitly in the affiliation block.
- Unambiguous Indian institution names/acronyms (safe to trust alone): IISc, IIIT, IIM, BITS Pilani, Anna University, Jadavpur University, Amrita, Manipal, DRDO, ISRO, CSIR, BARC, TIFR, IISER, Tata Research, Infosys Research, Wipro Research.
- STRICT RULE ON AMBIGUOUS ACRONYMS: "IIT", "NIT", "ISI", and "VIT" are NOT safe to trust alone. Each is also the real, in-use abbreviation for a well-known NON-Indian institution somewhere in the world — for example "IIT" is Istituto Italiano di Tecnologia in Italy (e.g. "IIT Genova"), and also the Institute of Informatics & Telecommunications at NCSR Demokritos in Greece; "ISI" is Istituto Superiore di Sanità in Italy. Do NOT mark an author Indian-affiliated from a bare "IIT"/"NIT"/"ISI"/"VIT" match by itself. Only count it as Indian if the SAME affiliation string also explicitly says "India", names an Indian city/state, or spells out the full name (e.g. "Indian Institute of Technology Bombay", "Vellore Institute of Technology") — otherwise leave that author out.
- STRICT RULE 1: Evaluate affiliations ONLY based on the official institutional affiliation strings (typically denoted by superscripts matching the author names). 
- STRICT RULE 2: Ignore email addresses entirely when determining affiliation. Do not infer institutional affiliation from email handles or domains (e.g., ignore "iitd" in "xyz.iitd@gmail.com").
- STRICT RULE 3: For multinational companies (e.g., General Motors, Google, Microsoft), do NOT assume an Indian affiliation unless an Indian city or "India" is explicitly stated in the affiliation string itself.
- STRICT RULE 4: Do NOT assume Indian affiliation from an author's name alone.
- STRICT RULE 5: Do NOT mark an affiliation Indian from a city name or company name alone unless the affiliation also contains an explicit Indian institution pattern or "India".
- STRICT RULE 6: When genuinely unsure whether an affiliation is Indian (e.g. it only matches an ambiguous acronym per the rule above, with no corroborating "India"/Indian city/full name), do NOT include that author — leaving an uncertain author out is always preferred over guessing.

PAGE CONTENT:
{content}

Return ONLY this JSON (no markdown, no explanation):
{{
  "paper_title": "exact title as it appears",
  "area_of_research": "one area from the list above",
  "total_authors": 0,
  "all_authors": [
    {{"name": "Full Name", "affiliation": "Full affiliation string or Unknown"}}
  ],
  "authors_with_indian_affiliations": ["Name1", "Name2"],
  "indian_institutions": ["Institution1", "Institution2"]
}}

If no Indian affiliations exist, use empty lists for the last two fields.
"""


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
                raw = self._call_llm(prompt + (retry_instruction if attempt else ""))
                try:
                    parsed = self._parse_json(raw)
                    break
                except Exception as e:
                    last_error = e

            if parsed is None:
                raise last_error or ValueError(f"No valid JSON found in LLM response: {raw[:200]}")

            info.paper_title = parsed.get("paper_title", "")
            info.area_of_research = parsed.get("area_of_research", "Others")
            info.total_authors = parsed.get("total_authors", 0)
            info.all_authors = parsed.get("all_authors", [])
            info.authors_with_indian_affiliations = parsed.get(
                "authors_with_indian_affiliations", []
            )
            info.indian_institutions = parsed.get("indian_institutions", [])

            # verify LLM didn't miss any Indian affiliations
            info = self._verify_indian_affiliations(info)

        except Exception as e:
            info.error = str(e)

        return info

    def _call_llm(self, prompt: str) -> str:
        """LLM call"""
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
        if self._should_request_json_mode():
            request["response_format"] = {"type": "json_object"}

        try:
            response = client.chat.completions.create(**request)
        except Exception:
            request.pop("response_format", None)
            response = client.chat.completions.create(**request)

        message = response.choices[0].message
        content = message.content or ""
        return content.strip()

    def _should_request_json_mode(self) -> bool:
        value = os.getenv("LLM_RESPONSE_FORMAT_JSON", "auto").lower()
        if value in {"1", "true", "yes"}:
            return True
        if value in {"0", "false", "no"}:
            return False
        return "groq.com" in self.base_url or self.provider in {"openai", "groq"}

    def _parse_json(self, raw: str) -> dict:
        """Extract JSON from LLM response robustly."""
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        # Find the outermost JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        raise ValueError(f"No valid JSON found in LLM response: {raw[:200]}")

    def _verify_indian_affiliations(self, info: PaperInfo) -> PaperInfo:
        """
        Two-way deterministic safety net around the LLM's own India-affiliation
        call, using evaluation.india_rules as ground truth:

        1. ADD authors the LLM missed whose affiliation is an unambiguous
           ("positive") Indian-institution match.
        2. DEMOTE (remove) authors the LLM flagged whose OWN affiliation
           string is only "ambiguous" — i.e. it matched purely on a
           collision-prone acronym (IIT/NIT/ISI/VIT — see india_rules.py's
           AMBIGUOUS_ACRONYM_INSTITUTION_PATTERNS) or a bare Indian
           city/company name, with no corroborating "India"/full name. The
           prompt already instructs the LLM not to do this, but a smaller
           model can still slip (this is exactly how "IIT" = Istituto
           Italiano di Tecnologia / a Greek NCSR Demokritos institute ended
           up misclassified as Indian in practice) — catch it
           deterministically rather than trusting the instruction alone.
        """
        if not info.all_authors:
            return info

        aff_by_name = {
            author.get("name", ""): author.get("affiliation", "")
            for author in info.all_authors
            if author.get("name")
        }

        missed_authors = []
        missed_institutions = set()

        for author in info.all_authors:
            aff = author.get("affiliation", "")
            if classify_affiliation(aff).label == "positive":
                name = author.get("name", "")
                if name not in info.authors_with_indian_affiliations:
                    missed_authors.append(name)
                    missed_institutions.add(aff)

        if missed_authors:
            info.authors_with_indian_affiliations.extend(missed_authors)
            info.indian_institutions.extend(list(missed_institutions))

        demoted = {
            name for name in info.authors_with_indian_affiliations
            if name in aff_by_name and classify_affiliation(aff_by_name[name]).label == "ambiguous"
        }
        if demoted:
            info.authors_with_indian_affiliations = [
                name for name in info.authors_with_indian_affiliations if name not in demoted
            ]
            # indian_institutions is free text (sometimes the LLM's own
            # cleaned-up name, sometimes a raw affiliation string), so we
            # can't always tie an entry back to the demoted author with
            # certainty. What we CAN safely drop is a bare ambiguous
            # acronym standing alone as an "institution" — e.g. "IIT" with
            # nothing else — which is itself the exact failure pattern
            # being corrected here, never a legitimate full institution name.
            info.indian_institutions = [
                inst for inst in info.indian_institutions
                if inst.strip().lower() not in {"iit", "nit", "isi", "vit"}
            ]

        return info
