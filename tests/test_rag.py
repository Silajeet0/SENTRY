import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kg import KnowledgeGraph
from rag import RAGEngine
from rag.embeddings import HashEmbedder
from benchmark.test_laya import load_dataset, metrics, validate_probabilities


class RAGTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.kg = KnowledgeGraph(self.path / "kg.sqlite3")
        self.csv = self.path / "orgs.csv"
        fields = ["entity_id", "canonical_name", "aliases", "country", "state", "city", "verification_status", "entity_type", "source_url"]
        self.records = [
            ["ORG_B", "Indian Institute of Technology Bombay", "IIT Bombay|IITB", "India", "Maharashtra", "Mumbai", "TRUSTED", "INSTITUTE", "https://iitb.ac.in"],
            ["ORG_J", "Jadavpur University", "Jadavpur Univ", "India", "West Bengal", "Kolkata", "TRUSTED", "UNIVERSITY", "https://jadavpuruniversity.in"],
            ["ORG_P", "Pending University", "Pending U", "India", "West Bengal", "Kolkata", "NEEDS_REVIEW", "UNIVERSITY", "https://example.org"],
        ]
        with self.csv.open("w", newline="") as f:
            writer = csv.writer(f); writer.writerow(fields); writer.writerows(self.records)
        self.kg.import_organizations(self.csv)
        self.paper = {"paper_url": "https://example.org/p1", "paper_title": "Video temporal grounding",
                      "abstract": "Locating moments in video with natural language queries.", "year": 2026,
                      "conference": "CVPR", "area_of_research": "Vision",
                      "all_authors": [{"name": "A Person", "affiliation": "Dept. CSE, IIT Bombay"},
                                      {"name": "B Person", "affiliation": "Jadavpur University"},
                                      {"name": "C Person", "affiliation": "Unknown Corporate Lab"}]}
        self.pid = self.kg.ingest_paper(self.paper)
        self.engine = RAGEngine(self.kg, HashEmbedder())

    def tearDown(self):
        self.kg.close()
        self.tmp.cleanup()

    def test_idempotent_import_ingestion_and_vectors(self):
        self.engine.index()
        before = (self.kg.version, len(self.kg.nodes()), len(self.kg.edges()))
        self.kg.import_organizations(self.csv)
        self.assertEqual(self.pid, self.kg.ingest_paper(self.paper))
        self.assertEqual(before, (self.kg.version, len(self.kg.nodes()), len(self.kg.edges())))
        self.assertEqual(0, self.engine.index())

    def test_unknowns_and_pending_records_never_become_trusted(self):
        candidates = self.kg.nodes("CANDIDATE")
        self.assertEqual(candidates[0]["verification_status"], "NEEDS_REVIEW")
        self.assertEqual("UNRESOLVED", self.kg.resolve("Pending U")["status"])
        self.engine.index()
        hits = self.engine.vector_search("Pending University", kind="ORGANIZATION")
        self.assertNotIn("ORG_P", [h["entity_id"] for h in hits])

    def test_alias_country_email_and_collision_guards(self):
        self.assertEqual("RESOLVED", self.kg.resolve("Dept CSE, IIT Bombay, Mumbai")["status"])
        for text in ("IIT", "IIT Bombay, Abu Dhabi, UAE", "x@iitb.ac.in", "NotIIT Bombay"):
            self.assertEqual("UNRESOLVED", self.kg.resolve(text)["status"], text)

    def test_pending_duplicate_cannot_add_trusted_alias(self):
        pending = self.path / "pending.csv"
        pending.write_text("entity_id,canonical_name,aliases,country,verification_status\nX,Jadavpur University,DangerAlias,India,NEEDS_REVIEW\n")
        self.kg.import_organizations(pending)
        self.assertNotIn("DangerAlias", self.kg.get("ORG_J")["aliases"])

    def test_location_update_removes_stale_edges(self):
        update = self.path / "update.csv"
        update.write_text("entity_id,canonical_name,country,state,city,verification_status\nORG_J,Jadavpur University,India,West Bengal,Updated City,TRUSTED\n")
        self.kg.import_organizations(update)
        locations = [self.kg.get(e["dst"]) for e in self.kg.edges("ORG_J") if e["relation"] == "LOCATED_IN"]
        self.assertIn("Updated City", [n["name"] for n in locations])
        self.assertNotIn("Kolkata", [n["name"] for n in locations])

    def test_paper_specific_author_identity_and_temporal_evidence(self):
        other = {**self.paper, "paper_url": "https://example.org/p2", "year": 2025,
                 "all_authors": [{"name": "A Person", "affiliation": "Jadavpur University"}]}
        pid = self.kg.ingest_paper(other)
        authors = [n for n in self.kg.nodes("AUTHOR") if n["name"] == "A Person"]
        self.assertEqual(2, len(authors))
        self.assertNotEqual(pid, self.pid)
        obs = [e for e in self.kg.edges() if e["relation"] == "AFFILIATED_WITH"]
        self.assertEqual({2025, 2026}, {e["data"]["observed_year"] for e in obs})

    def test_graph_collaboration_and_aggregate(self):
        hits = self.engine.search("Jadavpur University and IIT Bombay", mode="graph")
        self.assertEqual([self.pid], [h["entity_id"] for h in hits])
        self.assertEqual([], self.engine.search("IIT Bombay", mode="graph", filters={"year_max":2025}))
        counts = self.engine.aggregate({"country": "India"})
        self.assertEqual({"ORG_B", "ORG_J"}, {r["organization_id"] for r in counts})
        self.assertTrue(all(r["paper_count"] == 1 for r in counts))

    def test_filters_must_hold_on_one_organization(self):
        self.assertEqual([], self.engine.search("papers", mode="graph", filters={"organization_id":"ORG_B", "state":"West Bengal"}))
        with self.assertRaises(ValueError):
            self.engine.search("papers", mode="graph", filters={"made_up_filter":"x"})

    def test_hybrid_uses_graph_constraints_and_deduplicates(self):
        self.kg.ingest_paper({**self.paper, "paper_url":"https://example.org/p2", "paper_title":"Video video video",
                              "all_authors":[{"name":"D", "affiliation":"Elsewhere"}]})
        self.engine.index()
        results = self.engine.search("IIT Bombay video", k=10)
        self.assertEqual([self.pid], [r["entity_id"] for r in results])
        self.assertEqual(["graph", "vector"], results[0]["retrievers"])

    def test_stale_index_and_reingestion_remove_old_evidence(self):
        self.engine.index()
        self.kg.ingest_paper({**self.paper, "abstract": "Changed abstract", "all_authors": []})
        with self.assertRaisesRegex(ValueError, "stale"):
            self.engine.search("video", mode="vector")
        self.assertEqual(1, self.engine.index())
        self.assertFalse(any(e["relation"] == "AFFILIATED_WITH" for e in self.kg.paper_evidence(self.pid)["edges"]))

    def test_traversal_bounds_and_citations(self):
        with self.assertRaises(ValueError):
            self.kg.subgraph([self.pid], depth=6)
        self.assertLessEqual(len(self.kg.subgraph([self.pid], limit=2)["nodes"]),2)
        answer = self.engine.answer("IIT Bombay", mode="graph")
        self.assertIn("P1", answer["citations"])
        with self.assertRaisesRegex(ValueError,"citations"):
            self.engine.answer("IIT Bombay", mode="graph", generator=lambda _: "Made up [P99]")
        self.assertFalse(self.engine.answer("nothing matching", mode="graph")["citations"])

    def test_paths_and_same_name_author_lookup(self):
        self.assertTrue(self.kg.shortest_path("ORG_B", "ORG_J")["found"])
        self.assertFalse(self.kg.shortest_path("ORG_B", "ORG_J", max_depth=0)["found"])
        other = {**self.paper, "paper_url":"https://example.org/p2", "all_authors":[{"name":"A Person", "affiliation":"Jadavpur University"}]}
        self.kg.ingest_paper(other)
        self.assertEqual(2, len(self.engine.search("A Person publications", mode="graph")))

    def test_full_text_chunks_are_in_answer_evidence_and_removed_on_update(self):
        long = {**self.paper, "full_text": " ".join(["elephant"]*300)}
        self.kg.ingest_paper(long)
        self.engine.index()
        answer = self.engine.answer("elephant", mode="vector")
        self.assertIn("elephant", answer["evidence"][0]["retrieved_text"])
        self.kg.ingest_paper(self.paper)
        docs = self.kg.db.execute("SELECT metadata FROM documents WHERE entity_id=?", (self.pid,)).fetchall()
        self.assertEqual(["summary"], [json.loads(r[0])["section"] for r in docs])

    def test_changed_affiliations_retain_audit_history(self):
        self.kg.ingest_paper({**self.paper, "all_authors":[]})
        rows = self.kg.db.execute("SELECT previous FROM audit WHERE action='REPLACE_OBSERVATIONS' AND entity_id=? ORDER BY version DESC", (self.pid,)).fetchall()
        self.assertTrue(any(e["relation"] == "AFFILIATED_WITH" for e in json.loads(rows[0][0])))

    def test_ugc_schema_and_transaction_rollback(self):
        ugc = self.path / "ugc.csv"
        ugc.write_text('"Sr.No","Type","Name of the University","Address","Zip","state","Status"\n1,State,Example University,"Somewhere, Road",100000,Example,2(f)\n')
        self.assertEqual(1, self.kg.import_organizations(ugc))
        self.assertEqual("RESOLVED", self.kg.resolve("Example University")["status"])
        before = len(self.kg.nodes())
        broken = {**self.paper, "paper_url":"https://example.org/bad", "all_authors":[None]}
        with self.assertRaises(AttributeError):
            self.kg.ingest_paper(broken)
        self.assertEqual(before, len(self.kg.nodes()))

    def test_pipeline_hook_is_opt_in_and_preserves_errors(self):
        from extractors.llm_extractor import PaperInfo
        from rag.pipeline_hook import ingest
        with patch.dict(os.environ, {"SENTRY_RAG_ENABLED":"0"}), patch("kg.KnowledgeGraph") as store:
            ingest(PaperInfo("x", paper_title="x"), "CVPR", 2026)
            store.assert_not_called()
        with patch.dict(os.environ, {"SENTRY_RAG_ENABLED":"1", "SENTRY_KG_PATH": str(self.path/"hook.db")}):
            ingest(PaperInfo("x", paper_title="A real title"), "CVPR", 2026)
            graph = KnowledgeGraph(self.path/"hook.db")
            try:
                self.assertEqual(1, len(graph.nodes("PAPER")))
            finally:
                graph.close()


class BenchmarkTests(unittest.TestCase):
    def test_dataset_and_metrics(self):
        dataset = load_dataset(Path(__file__).resolve().parents[1]/"benchmark/benchmark_data.json")
        self.assertGreaterEqual(len(dataset["cases"]), 100)
        rows = [{"expected":"positive", "predicted":"positive"}, {"expected":"negative", "predicted":"ambiguous"}, {"expected":"ambiguous", "predicted":"positive"}]
        result = metrics(rows)
        self.assertAlmostEqual(result["accuracy"], 1/3)
        self.assertEqual(result["pipeline_inclusion"]["false_positive"],1)

    def test_probability_validation(self):
        p = validate_probabilities({"choice":"positive", "probabilities":dict.fromkeys(("positive","negative","ambiguous"),.3333)})
        self.assertAlmostEqual(sum(p.values()),1)
        with self.assertRaises(ValueError):
            validate_probabilities({"choice":"positive", "probabilities":{"positive":float("nan"),"negative":0,"ambiguous":0}})


if __name__ == "__main__":
    unittest.main()
