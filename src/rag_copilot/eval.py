"""Offline evaluation for the RAG pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import NO_EVIDENCE_MESSAGE
from .rag_chain import RAGChain, RAGResponse


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_keywords: list[str]
    expected_sources: list[str]
    expected_no_evidence: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        return cls(
            question=str(data["question"]),
            expected_keywords=[str(item) for item in data.get("expected_keywords", [])],
            expected_sources=[str(item) for item in data.get("expected_sources", [])],
            expected_no_evidence=bool(data.get("expected_no_evidence", False)),
        )


@dataclass(frozen=True)
class EvalResult:
    question: str
    answer: str
    expected_keywords: list[str]
    expected_sources: list[str]
    retrieved_sources: list[str]
    answer_correct: bool
    citation_correct: bool
    found_evidence: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "expected_keywords": self.expected_keywords,
            "expected_sources": self.expected_sources,
            "retrieved_sources": self.retrieved_sources,
            "answer_correct": self.answer_correct,
            "citation_correct": self.citation_correct,
            "found_evidence": self.found_evidence,
        }


@dataclass(frozen=True)
class EvalSummary:
    total: int
    answer_correct: int
    citation_correct: int

    @property
    def answer_accuracy(self) -> float:
        return self.answer_correct / self.total if self.total else 0.0

    @property
    def citation_accuracy(self) -> float:
        return self.citation_correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "answer_correct": self.answer_correct,
            "citation_correct": self.citation_correct,
            "answer_accuracy": self.answer_accuracy,
            "citation_accuracy": self.citation_accuracy,
        }


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    """Load JSONL evaluation cases."""

    cases: list[EvalCase] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        cases.append(EvalCase.from_dict(data))

    return cases


def evaluate(chain: RAGChain, cases: list[EvalCase]) -> tuple[list[EvalResult], EvalSummary]:
    """Run RAG answers and compute answer/citation accuracy."""

    results = [evaluate_case(chain, case) for case in cases]
    summary = EvalSummary(
        total=len(results),
        answer_correct=sum(result.answer_correct for result in results),
        citation_correct=sum(result.citation_correct for result in results),
    )
    return results, summary


def evaluate_case(chain: RAGChain, case: EvalCase) -> EvalResult:
    response = chain.answer(case.question)
    return score_response(case, response)


def score_response(case: EvalCase, response: RAGResponse) -> EvalResult:
    retrieved_sources = [citation.source for citation in response.citations]
    if case.expected_no_evidence:
        answer_correct = not response.found_evidence and NO_EVIDENCE_MESSAGE in response.answer
        citation_correct = not retrieved_sources
    else:
        answer_correct = _contains_all_keywords(response.answer, case.expected_keywords)
        citation_correct = _matches_expected_sources(retrieved_sources, case.expected_sources)

    return EvalResult(
        question=case.question,
        answer=response.answer,
        expected_keywords=case.expected_keywords,
        expected_sources=case.expected_sources,
        retrieved_sources=retrieved_sources,
        answer_correct=answer_correct,
        citation_correct=citation_correct,
        found_evidence=response.found_evidence,
    )


def write_eval_report(path: str | Path, results: list[EvalResult], summary: EvalSummary) -> Path:
    """Write a JSON report with per-case results and aggregate metrics."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary.to_dict(),
        "results": [result.to_dict() for result in results],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _contains_all_keywords(answer: str, expected_keywords: list[str]) -> bool:
    normalized_answer = _normalize(answer)
    return all(_normalize(keyword) in normalized_answer for keyword in expected_keywords)


def _matches_expected_sources(retrieved_sources: list[str], expected_sources: list[str]) -> bool:
    if not expected_sources:
        return True
    return all(
        any(_source_matches(actual, expected) for actual in retrieved_sources)
        for expected in expected_sources
    )


def _source_matches(actual: str, expected: str) -> bool:
    normalized_actual = actual.replace("\\", "/")
    normalized_expected = expected.replace("\\", "/")
    actual_name = normalized_actual.split("/")[-1]
    expected_name = normalized_expected.split("/")[-1]
    return (
        normalized_actual == normalized_expected
        or normalized_actual.startswith(f"{normalized_expected}#")
        or actual_name == expected_name
        or actual_name.startswith(f"{expected_name}#")
    )


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "")
