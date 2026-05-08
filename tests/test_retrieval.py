from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_copilot.models import Chunk
from rag_copilot.retriever import INDEX_FILE, VectorRetriever


class RetrievalTests(unittest.TestCase):
    def test_search_returns_relevant_chunk_with_source(self) -> None:
        chunks = [
            Chunk(
                id="refund",
                document_id="doc-1",
                source="faq.md",
                text="Customers can request a refund within 30 days. Receipts are required.",
                start_char=0,
                end_char=72,
                metadata={"title": "Refund policy"},
            ),
            Chunk(
                id="shipping",
                document_id="doc-2",
                source="shipping.md",
                text="Orders are shipped within 2 business days after payment is confirmed.",
                start_char=0,
                end_char=70,
                metadata={"title": "Shipping policy"},
            ),
        ]
        retriever = VectorRetriever.from_chunks(chunks)

        hits = retriever.search("refund receipt", top_k=2, min_score=0.0)

        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.id, "refund")
        self.assertEqual(hits[0].chunk.source, "faq.md")
        self.assertGreater(hits[0].score, 0)

    def test_search_returns_empty_when_no_chunk_reaches_threshold(self) -> None:
        chunks = [
            Chunk(
                id="refund",
                document_id="doc-1",
                source="faq.md",
                text="Customers can request a refund within 30 days.",
                start_char=0,
                end_char=48,
            )
        ]
        retriever = VectorRetriever.from_chunks(chunks)

        hits = retriever.search("unrelated quantum astronomy", min_score=0.5)

        self.assertEqual(hits, [])

    def test_index_can_be_saved_and_loaded(self) -> None:
        chunks = [
            Chunk(
                id="python",
                document_id="doc-1",
                source="skills.md",
                text="Python and SQL are listed as programming skills.",
                start_char=0,
                end_char=50,
            )
        ]
        retriever = VectorRetriever.from_chunks(chunks)

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = retriever.save(temp_dir)
            loaded = VectorRetriever.load(temp_dir)

            self.assertEqual(index_path.name, INDEX_FILE)
            self.assertEqual(len(loaded.chunks), 1)
            self.assertEqual(loaded.chunks[0].source, "skills.md")
            self.assertEqual(loaded.search("programming Python", min_score=0.0)[0].chunk.id, "python")


if __name__ == "__main__":
    unittest.main()
