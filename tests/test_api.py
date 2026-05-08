from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


if importlib.util.find_spec("fastapi") is None:
    raise unittest.SkipTest("FastAPI is not installed")

from rag_copilot import api
from rag_copilot.models import Chunk
from rag_copilot.retriever import VectorRetriever


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        api._retriever_cache = VectorRetriever.from_chunks(
            [
                Chunk(
                    id="skills-1",
                    document_id="doc-1",
                    source="skills.md",
                    text="Python and SQL are programming skills.",
                    start_char=0,
                    end_char=38,
                )
            ]
        )

    def tearDown(self) -> None:
        api._retriever_cache = None

    def test_health_returns_index_status(self) -> None:
        response = api.health()

        self.assertEqual(response.status, "ok")
        self.assertIsInstance(response.index_ready, bool)

    def test_ask_with_mock_returns_answer_and_citations(self) -> None:
        response = api.ask(
            api.AskRequest(
                question="Which programming skills are listed?",
                llm="mock",
                mock_response="Python and SQL are listed.[1]",
                top_k=1,
                min_score=0,
                show_prompt=True,
            )
        )

        self.assertEqual(response.answer, "Python and SQL are listed.[1]")
        self.assertTrue(response.found_evidence)
        self.assertEqual(response.citations[0].source, "skills.md")
        self.assertIsNotNone(response.prompt)
        self.assertIn("Python and SQL are programming skills.", response.prompt)


if __name__ == "__main__":
    unittest.main()
