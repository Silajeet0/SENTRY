import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.laya_classifier import (
    DEFAULT_LAYA_THRESHOLD,
    LABELS,
    LayaAffiliationClassifier,
    QUESTIONS,
    build_laya_request,
)


class LayaClassifierTests(unittest.TestCase):
    def test_validates_and_normalizes_laya_response(self):
        classifier = LayaAffiliationClassifier(model="unused")
        classifier._agent = SimpleNamespace(
            predict=lambda state, questions: {
                "answers": {
                    "affiliation_geography": {
                        "choice": "INDIAN",
                        "probabilities": {"INDIAN": 0.98, "NOT_INDIAN": 0.02},
                        "confidence": 0.8,
                    }
                }
            }
        )
        decision = classifier.predict("IIT Bombay, Mumbai")
        self.assertEqual(decision.label, "INDIAN")
        self.assertEqual(set(decision.probabilities), set(LABELS))
        self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0)
        self.assertEqual(decision.raw_label, "INDIAN")
        self.assertAlmostEqual(decision.max_probability, 0.98)
        self.assertAlmostEqual(decision.indian_probability, 0.98)
        self.assertEqual(decision.threshold, DEFAULT_LAYA_THRESHOLD)

    def test_threshold_maps_below_threshold_argmax_to_not_indian(self):
        classifier = LayaAffiliationClassifier(model="unused")
        classifier._agent = SimpleNamespace(
            predict=lambda *_: {
                "answers": {
                    "affiliation_geography": {
                        "choice": "INDIAN",
                        "probabilities": {"INDIAN": 0.9780, "NOT_INDIAN": 0.0220},
                        "confidence": 0.1,
                    }
                }
            }
        )
        decision = classifier.predict("Unresolved institution")
        self.assertEqual(decision.raw_label, "INDIAN")
        self.assertEqual(decision.label, "NOT_INDIAN")

    def test_threshold_accepts_boundary_probability(self):
        classifier = LayaAffiliationClassifier(model="unused")
        classifier._agent = SimpleNamespace(
            predict=lambda *_: {
                "answers": {
                    "affiliation_geography": {
                        "choice": "INDIAN",
                        "probabilities": {
                            "INDIAN": DEFAULT_LAYA_THRESHOLD,
                            "NOT_INDIAN": 1 - DEFAULT_LAYA_THRESHOLD,
                        },
                    }
                }
            }
        )
        self.assertEqual(classifier.predict("IIT Bombay, Mumbai").label, "INDIAN")

    def test_probabilities_are_required_for_thresholding(self):
        classifier = LayaAffiliationClassifier(model="unused")
        classifier._agent = SimpleNamespace(
            predict=lambda *_: {
                "answers": {"affiliation_geography": {"choice": "NOT_INDIAN"}}
            }
        )
        with self.assertRaises(ValueError):
            classifier.predict("Unknown affiliation")

    def test_rejects_invalid_choice(self):
        classifier = LayaAffiliationClassifier(model="unused")
        classifier._agent = SimpleNamespace(
            predict=lambda *_: {"answers": {"affiliation_geography": {"choice": "maybe"}}}
        )
        with self.assertRaises(ValueError):
            classifier.predict("Unknown affiliation")

    def test_questions_are_the_finetuning_contract(self):
        question = QUESTIONS["affiliation_geography"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(set(question["criteria"]), set(LABELS))

    def test_request_matches_laya_inference_shape(self):
        request = build_laya_request("IIT Bombay, Mumbai")
        self.assertEqual(set(request), {"state", "questions"})
        self.assertEqual(request["questions"], QUESTIONS)


class ExtractorLayaBoundaryTests(unittest.TestCase):
    def test_extractor_classifies_normalized_affiliations_with_laya(self):
        from extractors.llm_extractor import LLMExtractor

        payload = {
            "paper_title": "Graph learning",
            "abstract": "",
            "area_of_research": "Graph Analytics",
            "area_of_research_other": "",
            "total_authors": 99,
            "all_authors": [
                {"name": " A  Author ", "affiliation": " IIT Bombay,   Mumbai "},
                {"name": "Foreign Author", "affiliation": "Stanford University, USA"},
            ],
        }
        labels = iter(["INDIAN", "NOT_INDIAN"])
        with patch.object(LLMExtractor, "_call_llm", return_value=json.dumps(payload)), patch(
            "extractors.llm_extractor.classify_affiliation",
            side_effect=lambda _: SimpleNamespace(label=next(labels)),
        ):
            info = LLMExtractor().extract("source", "https://example.org/paper", "html")
        self.assertEqual(info.error, "")
        self.assertEqual(info.total_authors, 2)
        self.assertEqual(info.authors_with_indian_affiliations, ["A Author"])
        self.assertEqual(info.indian_institutions, ["IIT Bombay, Mumbai"])


if __name__ == "__main__":
    unittest.main()
