from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_copilot.eval import EvalCase, EvalSummary, load_eval_cases, score_response, write_eval_report
from rag_copilot.rag_chain import Citation, RAGResponse


class EvalTests(unittest.TestCase):
    def test_score_response_checks_answer_keywords_and_citation_sources(self) -> None:
        case = EvalCase(
            question="会哪些语言？",
            expected_keywords=["Python", "SQL"],
            expected_sources=["skills.md"],
        )
        response = RAGResponse(
            answer="他熟悉 Python 和 SQL。[1]",
            citations=[Citation(source="docs/skills.md", chunk_id="c1", score=0.7, start_char=0, end_char=20)],
            found_evidence=True,
        )

        result = score_response(case, response)

        self.assertTrue(result.answer_correct)
        self.assertTrue(result.citation_correct)
        self.assertEqual(result.retrieved_sources, ["docs/skills.md"])

    def test_score_response_supports_no_evidence_cases(self) -> None:
        case = EvalCase(question="不存在的问题", expected_keywords=[], expected_sources=[], expected_no_evidence=True)
        response = RAGResponse(answer="找不到依据", citations=[], found_evidence=False)

        result = score_response(case, response)

        self.assertTrue(result.answer_correct)
        self.assertTrue(result.citation_correct)

    def test_load_eval_cases_reads_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "questions.jsonl"
            path.write_text(
                '{"question":"Q1","expected_keywords":["A"],"expected_sources":["doc.md"]}\n',
                encoding="utf-8",
            )

            cases = load_eval_cases(path)

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].question, "Q1")
            self.assertEqual(cases[0].expected_keywords, ["A"])
            self.assertEqual(cases[0].expected_sources, ["doc.md"])

    def test_write_eval_report_outputs_summary_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.json"
            summary = EvalSummary(total=1, answer_correct=1, citation_correct=1)

            report_path = write_eval_report(path, [], summary)

            content = report_path.read_text(encoding="utf-8")
            self.assertIn('"answer_accuracy": 1.0', content)
            self.assertIn('"results": []', content)


if __name__ == "__main__":
    unittest.main()
