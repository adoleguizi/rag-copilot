"""Search the built local index and print matched chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_copilot.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, INDEX_DIR
from rag_copilot.retriever import VectorRetriever


def main() -> None:
    _configure_output_encoding()

    parser = argparse.ArgumentParser(description="Search the local RAG index.")
    parser.add_argument("query", help="Question or search query.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Maximum number of chunks to return.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum similarity score required to show a chunk.",
    )
    args = parser.parse_args()

    retriever = VectorRetriever.load(INDEX_DIR)
    hits = retriever.search(args.query, top_k=args.top_k, min_score=args.min_score)

    print(f"Query: {args.query}")
    print(f"Hits: {len(hits)}")

    if not hits:
        print("找不到依据")
        return

    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        print()
        print(f"[{index}] score={hit.score:.4f} source={chunk.source}")
        print(f"chunk_id={chunk.id} chars={chunk.start_char}-{chunk.end_char}")
        print(_preview(chunk.text))


def _preview(text: str, *, max_chars: int = 600) -> str:
    compact = "\n".join(line.rstrip() for line in text.strip().splitlines())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def _configure_output_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
