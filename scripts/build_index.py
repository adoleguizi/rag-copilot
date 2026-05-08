"""Build the local retrieval index from files in data/raw_docs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_copilot.chunking import chunk_documents
from rag_copilot.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, INDEX_DIR, RAW_DOCS_DIR
from rag_copilot.ingest import load_documents
from rag_copilot.retriever import VectorRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local RAG retrieval index.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Maximum characters per chunk.")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Characters shared between adjacent chunks.",
    )
    args = parser.parse_args()

    documents = load_documents(RAW_DOCS_DIR)
    chunks = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    retriever = VectorRetriever.from_chunks(chunks)
    index_path = retriever.save(INDEX_DIR)
    print(f"Loaded {len(documents)} documents")
    print(f"Built {len(chunks)} chunks")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Saved index to {index_path}")


if __name__ == "__main__":
    main()
