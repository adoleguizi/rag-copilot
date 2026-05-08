"""Shared data models for ingestion, chunking, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonValue = str | int | float | bool | None
Metadata = dict[str, JsonValue]


@dataclass(frozen=True)
class Document:
    """A loaded source document or source segment such as one PDF page."""

    id: str
    text: str
    source: str
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            source=str(data["source"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Chunk:
    """A searchable document chunk with source offsets for citation."""

    id: str
    document_id: str
    text: str
    source: str
    start_char: int
    end_char: int
    metadata: Metadata = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "text": self.text,
            "source": self.source,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=str(data["id"]),
            document_id=str(data["document_id"]),
            text=str(data["text"]),
            source=str(data["source"]),
            start_char=int(data["start_char"]),
            end_char=int(data["end_char"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval with its similarity score."""

    chunk: Chunk
    score: float
