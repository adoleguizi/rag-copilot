"""Small local vector retriever for the MVP RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K, DEFAULT_VECTOR_DIMENSION
from .models import Chunk, RetrievedChunk


INDEX_FILE = "vector_index.json"
WORD_RE = re.compile(r"[a-z0-9_]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

SparseVector = dict[int, float]


class VectorRetriever:
    """A dependency-free hashed TF-IDF vector retriever.

    This is intentionally simple for the first milestone. It gives us a stable
    local index, top-k retrieval, and citation-bearing chunks before adding an
    external embedding model or vector database.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        vectors: list[SparseVector],
        idf: dict[int, float],
        *,
        dimension: int = DEFAULT_VECTOR_DIMENSION,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors
        self.idf = idf
        self.dimension = dimension

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        *,
        dimension: int = DEFAULT_VECTOR_DIMENSION,
    ) -> "VectorRetriever":
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        hashed_counts = [_hashed_counts(chunk.text, dimension) for chunk in chunks]
        idf = _compute_idf(hashed_counts)
        vectors = [_normalize(_tfidf_vector(counts, idf)) for counts in hashed_counts]
        return cls(chunks=chunks, vectors=vectors, idf=idf, dimension=dimension)

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for a query."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        query_vector = self._query_vector(query)
        if not query_vector:
            return []

        scored: list[RetrievedChunk] = []
        for chunk, vector in zip(self.chunks, self.vectors, strict=True):
            score = _dot(query_vector, vector)
            if score >= min_score:
                scored.append(RetrievedChunk(chunk=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def save(self, index_dir: str | Path) -> Path:
        """Persist the retriever index as JSON."""

        target_dir = Path(index_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / INDEX_FILE
        payload = {
            "version": 1,
            "dimension": self.dimension,
            "idf": {str(key): value for key, value in self.idf.items()},
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "vectors": [_stringify_vector(vector) for vector in self.vectors],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorRetriever":
        """Load a retriever index from disk."""

        path = Path(index_dir) / INDEX_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(item) for item in data["chunks"]]
        vectors = [_parse_vector(item) for item in data["vectors"]]
        idf = {int(key): float(value) for key, value in data["idf"].items()}
        return cls(
            chunks=chunks,
            vectors=vectors,
            idf=idf,
            dimension=int(data.get("dimension", DEFAULT_VECTOR_DIMENSION)),
        )

    def _query_vector(self, query: str) -> SparseVector:
        counts = _hashed_counts(query, self.dimension)
        return _normalize(_tfidf_vector(counts, self.idf))


def tokenize(text: str) -> list[str]:
    """Tokenize English-like words and CJK text for retrieval."""

    lowered = text.lower()
    tokens = WORD_RE.findall(lowered)

    for match in CJK_RE.finditer(lowered):
        sequence = match.group(0)
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(max(len(sequence) - 1, 0)))

    return tokens


def _hashed_counts(text: str, dimension: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for token in tokenize(text):
        counts[_hash_token(token, dimension)] += 1
    return counts


def _hash_token(token: str, dimension: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def _compute_idf(doc_counts: list[Counter[int]]) -> dict[int, float]:
    total_docs = len(doc_counts)
    doc_frequency: Counter[int] = Counter()
    for counts in doc_counts:
        doc_frequency.update(counts.keys())
    return {
        index: math.log((1 + total_docs) / (1 + frequency)) + 1
        for index, frequency in doc_frequency.items()
    }


def _tfidf_vector(counts: Counter[int], idf: dict[int, float]) -> SparseVector:
    total_terms = sum(counts.values())
    if total_terms == 0:
        return {}
    return {
        index: (count / total_terms) * idf.get(index, 1.0)
        for index, count in counts.items()
    }


def _normalize(vector: SparseVector) -> SparseVector:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {index: value / norm for index, value in vector.items()}


def _dot(left: SparseVector, right: SparseVector) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def _stringify_vector(vector: SparseVector) -> dict[str, float]:
    return {str(key): value for key, value in vector.items()}


def _parse_vector(data: dict[str, Any]) -> SparseVector:
    return {int(key): float(value) for key, value in data.items()}
