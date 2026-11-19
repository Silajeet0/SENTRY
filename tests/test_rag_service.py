import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kg import KnowledgeGraph
from rag.corpus import sync_outputs
from rag.embeddings import HashEmbedder
from rag.engine import RAGEngine
from rag.service import query_knowledge, knowledge_status


class RAGServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "graph.db"
        self.env = patch.dict(os.environ, {"SENTRY_KG_PATH":str(self.path)})
        self.env.start()
        self.kg = KnowledgeGraph(self.path)
        self.paper = {"paper_url":"https://example.org/gnn", "paper_title":"Graph neural networks",
                      "abstract":"Graph neural networks learn graph representations.", "year":2026,
                      "all_authors":[{"name":"India Author", "affiliation":"IIT Bombay, India"},
                                     {"name":"Overseas Coauthor", "affiliation":"Stanford University, USA"}],
                      "authors_with_indian_affiliations":["India Author"]}
        self.pid = self.kg.ingest_paper(self.paper)
        RAGEngine(self.kg, HashEmbedder()).index()
        self.model = patch("rag.service.get_embedder", return_value=HashEmbedder())
        self.model.start()

    def tearDown(self):
        self.model.stop()
        self.env.stop()
        self.kg.close()
        self.temp.cleanup()

    def test_india_query_returns_only_observed_indian_authors(self):
        result = query_knowledge("Who works on GNN in India?", semantic_query="GNN", min_similarity=0)
        self.assertEqual(result["target"], "authors")
        self.assertEqual(result["semantic_query"], "graph neural networks")
        self.assertEqual([a["name"] for a in result["authors"]], ["India Author"])
        self.assertEqual(result["authors"][0]["geographic_evidence"], "sentry_pipeline_observation")
        self.assertEqual(result["citations"]["P1"]["url"], self.paper["paper_url"])
        strict = query_knowledge("Who works on GNN in India?", filters={"india_evidence":"trusted"}, min_similarity=0)
        self.assertEqual(strict["authors"], [])

    def test_from_india_phrase_applies_pipeline_geography_filter(self):
        result = query_knowledge(
            "Which authors from India work on reinforcement learning?",
            semantic_query="RL",
            min_similarity=0,
        )
        self.assertEqual(result["filters"], {"country": "India", "india_evidence": "pipeline"})
        self.assertEqual([a["name"] for a in result["authors"]], ["India Author"])
        self.assertEqual(result["semantic_query"], "reinforcement learning")

    def test_author_name_filter_returns_papers_and_observed_coauthors(self):
        self.kg.ingest_paper({
            **self.paper,
            "paper_url": "https://example.org/rl",
            "paper_title": "Reinforcement learning",
            "abstract": "Reinforcement learning methods.",
            "all_authors": [
                {"name": "Washim Uddin Mondal", "affiliation": "IIT Kanpur, India"},
                {"name": "Coauthor One", "affiliation": "Stanford University, USA"},
            ],
            "authors_with_indian_affiliations": ["Washim Uddin Mondal"],
        })
        RAGEngine(self.kg, HashEmbedder()).index()
        result = query_knowledge(
            "List the papers authored by Washim Uddin Mondal and his coauthors",
            semantic_query="reinforcement learning",
            target="papers",
            filters={"author_name": "Washim Uddin Mondal"},
            mode="hybrid",
            min_similarity=0,
        )
        self.assertEqual([p["title"] for p in result["papers"]], ["Reinforcement learning"])
        self.assertEqual(
            {a["name"] for a in result["authors"]},
            {"Washim Uddin Mondal", "Coauthor One"},
        )

    def test_author_only_lookup_does_not_apply_semantic_threshold(self):
        self.kg.ingest_paper({
            **self.paper,
            "paper_url": "https://example.org/washim-recent",
            "paper_title": "Recent author paper",
            "year": 2026,
            "all_authors": [{"name": "Washim Uddin Mondal", "affiliation": "IIT Kanpur, India"}],
        })
        RAGEngine(self.kg, HashEmbedder()).index()
        result = query_knowledge(
            "List some of Washim Uddin Mondal's recent papers",
            target="papers",
            filters={"author_name": "Washim Uddin Mondal"},
        )
        self.assertEqual(result["mode"], "graph")
        self.assertEqual([p["title"] for p in result["papers"]], ["Recent author paper"])

    def test_similarity_threshold_does_not_admit_graph_only_papers(self):
        result = query_knowledge("Who works on quantum fermentation in India?", min_similarity=1)
        self.assertEqual(result["papers"], [])

    def test_semantic_query_preserves_original_named_entity_constraint(self):
        self.kg.ingest_paper({**self.paper,"conference":"TESTCONF"})
        self.kg.ingest_paper({**self.paper,"paper_url":"https://example.org/other","conference":"OTHERCONF"})
        RAGEngine(self.kg,HashEmbedder()).index()
        result = query_knowledge("Who works on GNN at TESTCONF in India?",semantic_query="GNN",min_similarity=0)
        self.assertEqual(len(result["papers"]),1)
        self.assertEqual(result["papers"][0]["conference"],"TESTCONF")

    def test_multiple_years_of_conference_are_alternative_event_nodes(self):
        self.kg.ingest_paper({**self.paper,"conference":"TESTCONF"})
        self.kg.ingest_paper({**self.paper,"paper_url":"https://example.org/prior","conference":"TESTCONF","year":2025})
        result = query_knowledge("TESTCONF papers",mode="graph")
        self.assertEqual(len(result["papers"]),2)

    def test_trusted_geography_never_spreads_to_foreign_coauthors(self):
        csv = self.root/"org.csv"
        csv.write_text("entity_id,canonical_name,aliases,country,state,verification_status\nORG_X,Indian Institute of Technology Bombay,IIT Bombay,India,Maharashtra,TRUSTED\n")
        self.kg.import_organizations(csv)
        self.kg.ingest_paper(self.paper)
        RAGEngine(self.kg,HashEmbedder()).index()
        result = query_knowledge("Who works on GNN in India?", filters={"india_evidence":"trusted"}, min_similarity=0)
        self.assertEqual([a["name"] for a in result["authors"]], ["India Author"])
        self.assertEqual(result["authors"][0]["geographic_evidence"],"trusted_organization")

    def test_sync_preserves_versions_conflicts_and_reports_incomplete_rows(self):
        for conference, year in (("KDD","2026v1"),("OTHER","2026")):
            path = self.root/"outputs"/conference/year
            path.mkdir(parents=True)
            (path/"indian_papers_structured.json").write_text(json.dumps([self.paper]))
        bad = self.root/"outputs"/"BAD"/"2025"
        bad.mkdir(parents=True)
        (bad/"indian_papers_structured.json").write_text(json.dumps([{**self.paper,"paper_url":"https://example.org/missing","paper_title":""}]))
        report = sync_outputs(self.kg, self.root/"outputs")
        self.assertEqual(report["source_records"],3)
        self.assertEqual(report["ingested_unique_papers"],1)
        self.assertEqual(len(report["errors"]),1)
        self.assertEqual(report["duplicate_records"],1)
        paper = self.kg.get(self.pid)
        self.assertEqual(paper["year"],2026)
        self.assertEqual(paper["conference"],"")
        self.assertEqual(len(paper["raw_record"]["corpus_sources"]),2)
        version = self.kg.version
        sync_outputs(self.kg, self.root/"outputs")
        self.assertEqual(version,self.kg.version)
        self.assertEqual(knowledge_status()["corpus_imports"][0]["ingested_unique_papers"],1)

    def test_sync_enriches_from_exact_url_abstract_cache(self):
        folder = self.root/"outputs"/"CONF"/"2026"
        folder.mkdir(parents=True)
        record = {**self.paper,"abstract":""}
        (folder/"indian_papers_structured.json").write_text(json.dumps([record]))
        (folder/"abstracts_cache.json").write_text(json.dumps({self.paper["paper_url"]:{"paper_url":self.paper["paper_url"],"abstract":"Cached graph learning evidence"}}))
        report = sync_outputs(self.kg,self.root/"outputs")
        paper = self.kg.get(self.pid)
        self.assertEqual(paper["abstract"],"Cached graph learning evidence")
        self.assertIn("sha256",paper["raw_record"]["abstract_cache_source"])
        self.assertEqual(report["missing_abstracts"],0)

    def test_http_stale_index_returns_actionable_conflict(self):
        from fastapi.testclient import TestClient
        import orchestrator_api
        self.kg.ingest_paper({**self.paper,"abstract":"Changed text"})
        with patch.object(orchestrator_api,"_API_KEYS",set()):
            response = TestClient(orchestrator_api.app).post('/v1/rag/query',json={"question":"GNN"})
        self.assertEqual(response.status_code,409)
        self.assertIn('index',response.json()['detail'])

    def test_http_auth_validation_and_real_service(self):
        from fastapi.testclient import TestClient
        import orchestrator_api
        with patch.object(orchestrator_api, "_API_KEYS", {"test-secret"}):
            client = TestClient(orchestrator_api.app)
            self.assertEqual(client.get("/v1/rag/status").status_code,401)
            headers = {"Authorization":"Bearer test-secret"}
            response = client.post("/v1/rag/query",headers=headers,json={"question":"Who works on GNN in India?", "min_similarity":0})
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()["authors"][0]["name"],"India Author")
            self.assertEqual(client.post("/v1/rag/query",headers=headers,json={"question":"x", "k":50}).status_code,422)
            self.assertEqual(client.post("/v1/rag/query",headers=headers,json={"question":"x", "filters":{"unknown":True}}).status_code,422)
            self.assertEqual(client.get("/v1/rag/status",headers=headers).status_code,200)

    def test_orchestrator_dispatches_rag_tool_and_receives_citations(self):
        from orchestrator.agent import Orchestrator
        from orchestrator.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS
        self.assertIn("query_scholarly_knowledge",TOOL_FUNCTIONS)
        self.assertIn("query_scholarly_knowledge",{s["function"]["name"] for s in TOOL_SCHEMAS})
        call = SimpleNamespace(id="rag1", function=SimpleNamespace(name="query_scholarly_knowledge",arguments=json.dumps({"question":"Who works on GNN in India?","min_similarity":0})))
        messages = [SimpleNamespace(content=None,tool_calls=[call]), SimpleNamespace(content="India Author [P1]",tool_calls=None)]
        agent = Orchestrator()
        with patch.object(agent,"_client"),patch.object(agent,"_call_llm",side_effect=[SimpleNamespace(choices=[SimpleNamespace(message=m)]) for m in messages]):
            self.assertEqual(agent.chat("Who works on GNN in India?"),"India Author [P1]")
        tool = json.loads(next(m["content"] for m in agent.messages if m["role"]=="tool"))
        self.assertEqual(tool["authors"][0]["name"],"India Author")
        self.assertIn("P1",tool["citations"])


if __name__ == "__main__":
    unittest.main()
