"""Run offline RAG evaluation against data/eval/questions.jsonl."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_copilot.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, EVAL_DIR, INDEX_DIR
from rag_copilot.eval import evaluate, load_eval_cases, write_eval_report
from rag_copilot.llm import create_model
from rag_copilot.rag_chain import RAGChain
from rag_copilot.retriever import INDEX_FILE, VectorRetriever


def main() -> int:
    _configure_output_encoding()
    _load_env_file(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run offline RAG evaluation.")
    parser.add_argument("--eval-file", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--results-file", type=Path, default=EVAL_DIR / "results.json")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--llm", choices=["mock", "dashscope"], default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument(
        "--mock-response",
        default="孙志浩熟悉 C/C++、Python、SQL 和 shell 脚本。[1]",
        help="Deterministic answer returned by MockLLM.",
    )
    parser.add_argument("--show-details", action="store_true")
    args = parser.parse_args()

    index_path = args.index_dir / INDEX_FILE
    if not index_path.exists():
        print(f"Index not found: {index_path}", file=sys.stderr)
        print("Run this first: .\\venv\\Scripts\\python.exe scripts\\build_index.py", file=sys.stderr)
        return 1

    if not args.eval_file.exists():
        print(f"Eval file not found: {args.eval_file}", file=sys.stderr)
        return 1

    retriever = VectorRetriever.load(args.index_dir)
    chain = RAGChain(
        retriever,
        _build_llm(args),
        top_k=args.top_k,
        min_score=args.min_score,
    )

    try:
        cases = load_eval_cases(args.eval_file)
        results, summary = evaluate(chain, cases)
    except RuntimeError as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    report_path = write_eval_report(args.results_file, results, summary)
    _print_summary(summary.total, summary.answer_accuracy, summary.citation_accuracy, report_path)

    if args.show_details:
        _print_details(results)

    return 0


def _build_llm(args: argparse.Namespace):
    return create_model(
        args.llm,
        mock_response=args.mock_response,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def _print_summary(total: int, answer_accuracy: float, citation_accuracy: float, report_path: Path) -> None:
    print(f"Total: {total}")
    print(f"Answer accuracy: {answer_accuracy:.2%}")
    print(f"Citation accuracy: {citation_accuracy:.2%}")
    print(f"Saved report to {report_path}")


def _print_details(results: list[object]) -> None:
    for index, result in enumerate(results, start=1):
        print()
        print(f"[{index}] {result.question}")
        print(f"answer_correct={result.answer_correct} citation_correct={result.citation_correct}")
        print(f"retrieved_sources={result.retrieved_sources}")
        print(f"answer={result.answer}")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
