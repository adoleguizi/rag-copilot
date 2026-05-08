"""RAG orchestration without binding to a specific LLM provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K
from .models import RetrievedChunk
from .prompts import NO_EVIDENCE_MESSAGE, build_rag_prompt


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int, min_score: float) -> list[RetrievedChunk]:
        """Return relevant chunks for a query."""


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate an answer from a completed prompt."""


@dataclass(frozen=True)
class Citation:
    source: str
    chunk_id: str
    score: float
    start_char: int
    end_char: int

    @classmethod
    def from_hit(cls, hit: RetrievedChunk) -> "Citation":
        chunk = hit.chunk
        return cls(
            source=chunk.source,
            chunk_id=chunk.id,
            score=hit.score,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    citations: list[Citation]
    found_evidence: bool
    prompt: str | None = None

    def to_text(self) -> str:
        if not self.found_evidence:
            return self.answer

        citation_lines = [
            f"[{index}] {citation.source} chars={citation.start_char}-{citation.end_char} "
            f"score={citation.score:.4f}"
            for index, citation in enumerate(self.citations, start=1)
        ]
        return f"{self.answer}\n\n引用来源：\n" + "\n".join(citation_lines)


class RAGChain:
    """Retrieve evidence, build a grounded prompt, and call an LLM client."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if min_score < 0:
            raise ValueError("min_score must be greater than or equal to 0")

        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def answer(self, question: str) -> RAGResponse:
        hits = self.retriever.search(question, top_k=self.top_k, min_score=self.min_score)
        if not hits:
            return RAGResponse(answer=NO_EVIDENCE_MESSAGE, citations=[], found_evidence=False)

        prompt = build_rag_prompt(question, hits)
        answer = self.llm.generate(prompt).strip() or NO_EVIDENCE_MESSAGE
        citations = [Citation.from_hit(hit) for hit in hits]
        return RAGResponse(answer=answer, citations=citations, found_evidence=True, prompt=prompt)


class MockLLM:
    """Deterministic fake LLM for tests and local flow validation."""

    def __init__(self, response: str = "这是一个测试答案。[1]") -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class EchoLLM:
    """Fake LLM that returns the prompt unchanged for prompt inspection."""

    def generate(self, prompt: str) -> str:
        return prompt
