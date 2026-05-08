from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_copilot.chunking import chunk_documents, chunk_text
from rag_copilot.models import Document


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_splits_long_text_and_preserves_offsets(self) -> None:
        text = (
            "Refund policy allows refunds within 30 days. "
            "Receipts are required for every refund request.\n\n"
            "Shipping policy says orders usually ship within 2 business days. "
            "Tracking numbers are sent by email.\n\n"
            "Support policy says urgent requests should use the emergency contact channel."
        )

        spans = chunk_text(text, chunk_size=120, chunk_overlap=20)

        self.assertGreater(len(spans), 1)
        for span in spans:
            self.assertLessEqual(len(span.text), 120)
            self.assertEqual(span.text, text[span.start_char : span.end_char])
            self.assertLess(span.start_char, span.end_char)

        starts = [span.start_char for span in spans]
        self.assertEqual(starts, sorted(starts))

    def test_chunk_documents_keeps_source_and_metadata(self) -> None:
        document = Document(
            id="doc-1",
            source="faq.md",
            text="Question: What is the refund window?\nAnswer: Refunds are allowed within 30 days.",
            metadata={"title": "Refund FAQ", "file_type": "md"},
        )

        chunks = chunk_documents([document], chunk_size=60, chunk_overlap=10)

        self.assertGreaterEqual(len(chunks), 1)
        first_chunk = chunks[0]
        self.assertEqual(first_chunk.document_id, "doc-1")
        self.assertEqual(first_chunk.source, "faq.md")
        self.assertEqual(first_chunk.metadata["title"], "Refund FAQ")
        self.assertEqual(first_chunk.metadata["file_type"], "md")
        self.assertEqual(first_chunk.metadata["chunk_index"], 0)

    def test_chunk_text_prefers_generic_heading_boundaries(self) -> None:
        text = (
            "Overview\n"
            "This product helps teams search internal knowledge.\n\n"
            "Capabilities\n"
            "Searches markdown files.\n"
            "Returns citations.\n"
            "Runs offline evaluation.\n\n"
            "Limitations\n"
            "It should not answer without evidence."
        )

        spans = chunk_text(text, chunk_size=200, chunk_overlap=30)
        capability_span = next(span for span in spans if "Capabilities" in span.text)

        self.assertIn("Returns citations", capability_span.text)
        self.assertNotIn("Limitations", capability_span.text)

    def test_single_blank_line_before_short_heading_starts_new_section(self) -> None:
        text = (
            "Name\tRole\n"
            "Alice\tEngineer\n\n"
            "Skills\n"
            "Python\tAdvanced\n"
            "SQL\tAdvanced"
        )

        spans = chunk_text(text, chunk_size=200, chunk_overlap=30)
        skills_span = next(span for span in spans if span.text.startswith("Skills"))

        self.assertIn("SQL", skills_span.text)
        self.assertNotIn("Alice", skills_span.text)

    def test_colon_heading_prefix_starts_new_section(self) -> None:
        text = (
            "Campus activity\n"
            "Volunteered for student events.\n\n"
            "Work Experience: Example Studio role: Unity intern dates: 2025-01 to 2025-02\n"
            "Built UI animations.\n"
            "Integrated media playback.\n"
            "Second Company role: operations intern dates: 2025-02 to 2025-03\n"
            "Maintained deployment scripts."
        )

        spans = chunk_text(text, chunk_size=450, chunk_overlap=80)
        work_span = next(span for span in spans if span.text.startswith("Work Experience:"))

        self.assertIn("Second Company", work_span.text)
        self.assertNotIn("Campus activity", work_span.text)

    def test_chunk_text_preserves_markdown_sections(self) -> None:
        text = (
            "# Refund Policy\n"
            "Refunds are available within 30 days.\n\n"
            "## Shipping Policy\n"
            "Orders usually ship within 2 business days."
        )

        spans = chunk_text(text, chunk_size=200, chunk_overlap=30)

        self.assertEqual(len(spans), 2)
        self.assertTrue(spans[0].text.startswith("# Refund Policy"))
        self.assertTrue(spans[1].text.startswith("## Shipping Policy"))

    def test_invalid_chunk_settings_raise_clear_errors(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=0)

        with self.assertRaises(ValueError):
            chunk_text("hello", chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
