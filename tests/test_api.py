from __future__ import annotations

import importlib.util
import sys
import tempfile
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

    def test_list_documents_reads_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_raw_docs_dir = api.RAW_DOCS_DIR
            api.RAW_DOCS_DIR = Path(temp_dir)
            try:
                (api.RAW_DOCS_DIR / "faq.md").write_text("FAQ content", encoding="utf-8")
                (api.RAW_DOCS_DIR / "ignored.exe").write_text("nope", encoding="utf-8")

                documents = api.list_documents()

                self.assertEqual([document.name for document in documents], ["faq.md"])
            finally:
                api.RAW_DOCS_DIR = original_raw_docs_dir

    def test_rebuild_index_updates_retriever_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_raw_docs_dir = api.RAW_DOCS_DIR
            original_index_dir = api.INDEX_DIR
            api.RAW_DOCS_DIR = Path(temp_dir) / "raw"
            api.INDEX_DIR = Path(temp_dir) / "index"
            api.RAW_DOCS_DIR.mkdir()
            (api.RAW_DOCS_DIR / "knowledge.md").write_text(
                "# Skills\nPython and SQL are listed skills.",
                encoding="utf-8",
            )

            try:
                response = api.rebuild_index(api.RebuildIndexRequest(chunk_size=200, chunk_overlap=20))

                self.assertEqual(response.documents, 1)
                self.assertGreaterEqual(response.chunks, 1)
                self.assertIsNotNone(api._retriever_cache)
                self.assertTrue((api.INDEX_DIR / "vector_index.json").exists())
            finally:
                api.RAW_DOCS_DIR = original_raw_docs_dir
                api.INDEX_DIR = original_index_dir
                api._retriever_cache = None


if __name__ == "__main__":
    unittest.main()
