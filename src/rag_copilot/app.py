"""Minimal command-line RAG QA entrypoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, INDEX_DIR, PROJECT_ROOT
from .llm import create_model
from .rag_chain import RAGChain
from .retriever import INDEX_FILE, VectorRetriever


def main(argv: list[str] | None = None) -> int:
    _configure_output_encoding()
    _load_env_file(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Ask a question against the local RAG index.")
    parser.add_argument("question", help="Question to ask.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Maximum chunks used as evidence.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum retrieval score required to use a chunk as evidence.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
        help="Directory containing vector_index.json.",
    )
    parser.add_argument(
        "--mock-response",
        default="这是一个 MockLLM 测试答案。[1]",
        help="Deterministic answer returned by MockLLM.",
    )
    parser.add_argument(
        "--llm",
        choices=["mock", "dashscope"],
        default="mock",
        help="LLM backend to use. Defaults to mock for offline testing.",
    )
    parser.add_argument("--model", default=None, help="Model name for --llm dashscope.")
    parser.add_argument("--base-url", default=None, help="Base URL for --llm dashscope.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=800, help="Maximum output tokens for DashScope.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the generated prompt after the answer.")
    args = parser.parse_args(argv)

    index_path = args.index_dir / INDEX_FILE
    if not index_path.exists():
        print(f"Index not found: {index_path}", file=sys.stderr)
        print("Run this first: .\\venv\\Scripts\\python.exe scripts\\build_index.py", file=sys.stderr)
        return 1

    retriever = VectorRetriever.load(args.index_dir)
    llm = _build_llm(args)
    chain = RAGChain(
        retriever,
        llm,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    try:
        response = chain.answer(args.question)
    except RuntimeError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 1

    print(f"问题：{args.question}")
    print()
    print(response.to_text())

    if args.show_prompt and response.prompt:
        print()
        print("生成的 Prompt：")
        print(response.prompt)

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
