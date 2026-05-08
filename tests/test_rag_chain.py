from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_copilot.models import Chunk, RetrievedChunk
from rag_copilot.prompts import NO_EVIDENCE_MESSAGE
from rag_copilot.rag_chain import MockLLM, RAGChain


class StaticRetriever:
    def __init__(self, hits: list[RetrievedChunk]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, float]] = []

    def search(self, query: str, *, top_k: int, min_score: float) -> list[RetrievedChunk]:
        self.calls.append((query, top_k, min_score))
        return self.hits[:top_k]


class RAGChainTests(unittest.TestCase):
    def test_answer_calls_llm_with_retrieved_context_and_returns_citations(self) -> None:
        hit = RetrievedChunk(
            chunk=Chunk(
                id="skills-1",
                document_id="doc-1",
                source="skills.md",
                text="孙志浩熟悉 C/C++、Python、SQL 和 shell 脚本。",
                start_char=10,
                end_char=43,
                metadata={"title": "技能"},
            ),
            score=0.72,
        )
        retriever = StaticRetriever([hit])
        llm = MockLLM("孙志浩熟悉 C/C++、Python、SQL 和 shell 脚本。[1]")
        chain = RAGChain(retriever, llm, top_k=1, min_score=0.1)

        response = chain.answer("孙志浩会哪些编程语言？")

        self.assertTrue(response.found_evidence)
        self.assertEqual(response.answer, "孙志浩熟悉 C/C++、Python、SQL 和 shell 脚本。[1]")
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].source, "skills.md")
        self.assertEqual(response.citations[0].chunk_id, "skills-1")
        self.assertEqual(retriever.calls, [("孙志浩会哪些编程语言？", 1, 0.1)])
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("孙志浩熟悉 C/C++、Python、SQL 和 shell 脚本。", llm.prompts[0])
        self.assertIn("source=skills.md", llm.prompts[0])

    def test_answer_returns_no_evidence_and_does_not_call_llm_when_no_hits(self) -> None:
        retriever = StaticRetriever([])
        llm = MockLLM("不应该被调用")
        chain = RAGChain(retriever, llm, top_k=3, min_score=0.2)

        response = chain.answer("资料里不存在的问题")

        self.assertFalse(response.found_evidence)
        self.assertEqual(response.answer, NO_EVIDENCE_MESSAGE)
        self.assertEqual(response.citations, [])
        self.assertEqual(llm.prompts, [])

    def test_response_to_text_includes_citation_sources(self) -> None:
        hit = RetrievedChunk(
            chunk=Chunk(
                id="refund-1",
                document_id="doc-1",
                source="faq.md",
                text="退款期限是 30 天。",
                start_char=0,
                end_char=9,
            ),
            score=0.8,
        )
        chain = RAGChain(StaticRetriever([hit]), MockLLM("退款期限是 30 天。[1]"), top_k=1, min_score=0)

        text = chain.answer("退款期限是多少？").to_text()

        self.assertIn("退款期限是 30 天。[1]", text)
        self.assertIn("引用来源：", text)
        self.assertIn("[1] faq.md chars=0-9 score=0.8000", text)


if __name__ == "__main__":
    unittest.main()
