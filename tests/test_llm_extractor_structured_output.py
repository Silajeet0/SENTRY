import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from extractors.llm_extractor import EXTRACTION_JSON_SCHEMA, LLMExtractor


class _FakeCompletions:
    def __init__(self, captured):
        self.captured = captured

    def create(self, **kwargs):
        self.captured.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"paper_title":"x"}'))]
        )


class _FakeOpenAI:
    captured = []

    def __init__(self, **_kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions(self.captured))


class StructuredExtractionTests(unittest.TestCase):
    def setUp(self):
        _FakeOpenAI.captured.clear()
        self.openai_module = SimpleNamespace(OpenAI=_FakeOpenAI)

    def _call_for(self, provider, base_url):
        env = {
            "LLM_PROVIDER": provider,
            "LLM_MODEL": "test-model",
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": base_url,
            "LLM_RESPONSE_FORMAT_JSON": "auto",
        }
        with patch.dict(os.environ, env, clear=False), patch.dict(
            sys.modules, {"openai": self.openai_module}
        ):
            LLMExtractor()._call_llm("extract", EXTRACTION_JSON_SCHEMA)
        return _FakeOpenAI.captured[-1]

    def test_groq_uses_strict_json_schema(self):
        request = self._call_for("groq", "https://api.groq.com/openai/v1")

        response_format = request["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["schema"], EXTRACTION_JSON_SCHEMA)

    def test_nvidia_uses_guided_json(self):
        request = self._call_for("nvidia", "https://integrate.api.nvidia.com/v1")

        self.assertEqual(
            request["extra_body"], {"nvext": {"guided_json": EXTRACTION_JSON_SCHEMA}}
        )
        self.assertNotIn("response_format", request)

    def test_ollama_uses_native_schema_format_in_auto_mode(self):
        request = self._call_for("ollama", "http://localhost:11434/v1")

        self.assertEqual(request["extra_body"], {"format": EXTRACTION_JSON_SCHEMA})
        self.assertNotIn("response_format", request)

    def test_schema_is_strict_compatible(self):
        self.assertFalse(EXTRACTION_JSON_SCHEMA["additionalProperties"])
        author_schema = EXTRACTION_JSON_SCHEMA["properties"]["all_authors"]["items"]
        self.assertFalse(author_schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
