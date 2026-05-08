"""Chunk documents into citation-friendly text spans."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .models import Chunk, Document


@dataclass(frozen=True)
class TextSpan:
    text: str
    start_char: int
    end_char: int


BREAKPOINTS = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    "\u3002",
    "\uff1f",
    "\uff01",
    "\uff1b",
    "\uff0c",
)

MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[.)]|[A-Z][.)]|[IVXLCDM]+[.)]|[\u4e00-\u9fff]{1,4}[\u3001.．])\s*\S"
)
FAQ_QUESTION_RE = re.compile(r"^\s*(?:q|question|\u95ee)\s*[:\uff1a]\s*\S", re.IGNORECASE)
SENTENCE_ENDINGS = (".", "?", "!", ";", "\u3002", "\uff1f", "\uff01", "\uff1b")
BULLET_PREFIXES = ("-", "*", "+", "\u2022")


def chunk_documents(
    documents: list[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split loaded documents into chunks while preserving source metadata."""

    chunks: list[Chunk] = []
    for document in documents:
        spans = chunk_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for index, span in enumerate(spans):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            chunks.append(
                Chunk(
                    id=_chunk_id(document.id, span.start_char, span.end_char),
                    document_id=document.id,
                    text=span.text,
                    source=document.source,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    metadata=metadata,
                )
            )
    return chunks


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextSpan]:
    """Split text using generic structure before falling back to windows.

    The splitter does not depend on business-specific headings. It recognizes
    common document structure such as Markdown headings, numbered headings,
    FAQ questions, short title-like lines, blank paragraphs, and table rows.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return []

    sections = _split_sections(normalized)
    if len(sections) <= 1:
        return _chunk_window(
            normalized,
            0,
            len(normalized),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    spans: list[TextSpan] = []
    for section in sections:
        if len(section.text) <= chunk_size:
            spans.append(section)
            continue

        spans.extend(
            _chunk_window(
                normalized,
                section.start_char,
                section.end_char,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return spans


def _split_sections(text: str) -> list[TextSpan]:
    """Split text into coarse logical sections."""

    line_spans = _iter_line_spans(text)
    section_starts = _find_section_starts(line_spans)
    if len(section_starts) <= 1:
        return _split_paragraph_sections(text)

    if section_starts[0] != 0:
        section_starts.insert(0, 0)

    sections: list[TextSpan] = []
    for index, start in enumerate(section_starts):
        end = section_starts[index + 1] if index + 1 < len(section_starts) else len(text)
        trimmed_start, trimmed_end = _trim_span(text, start, end)
        if trimmed_start < trimmed_end:
            sections.append(TextSpan(text[trimmed_start:trimmed_end], trimmed_start, trimmed_end))
    return sections


def _iter_line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        spans.append((start, end, line.rstrip("\n")))
        start = end
    if start < len(text):
        spans.append((start, len(text), text[start:]))
    return spans


def _find_section_starts(line_spans: list[tuple[int, int, str]]) -> list[int]:
    starts: list[int] = []
    for index, (start, _end, line) in enumerate(line_spans):
        stripped = line.strip()
        if not stripped:
            continue

        previous = _previous_nonblank_index(line_spans, index)
        previous_blank = previous is None or previous < index - 1

        next_index = _next_nonblank_index(line_spans, index)
        next_line = line_spans[next_index][2].strip() if next_index is not None else ""

        if _is_structural_start(stripped, previous_blank=previous_blank, next_line=next_line):
            starts.append(start)
    return starts


def _is_structural_start(line: str, *, previous_blank: bool, next_line: str) -> bool:
    if _is_table_line(line):
        return False
    if MARKDOWN_HEADING_RE.match(line):
        return True
    if FAQ_QUESTION_RE.match(line):
        return True
    if previous_blank and _is_numbered_heading(line):
        return True
    if previous_blank and _looks_like_colon_heading_prefix(line):
        return True
    if previous_blank and next_line and _looks_like_short_heading(line):
        return True
    return False


def _is_numbered_heading(line: str) -> bool:
    if len(line) > 90 or line.endswith(SENTENCE_ENDINGS):
        return False
    return bool(NUMBERED_HEADING_RE.match(line))


def _looks_like_short_heading(line: str) -> bool:
    if len(line) > 60:
        return False
    if line.endswith(SENTENCE_ENDINGS):
        return False
    if line.startswith(BULLET_PREFIXES):
        return False
    if _is_table_line(line):
        return False
    if len(line.split()) > 8:
        return False
    return True


def _looks_like_colon_heading_prefix(line: str) -> bool:
    heading, separator, _rest = line.partition(":")
    if not separator:
        heading, separator, _rest = line.partition("\uff1a")
    if not separator:
        return False

    heading = heading.strip()
    if not heading or len(heading) > 30:
        return False
    if heading.startswith(BULLET_PREFIXES):
        return False
    if len(heading.split()) > 6:
        return False
    return True


def _is_table_line(line: str) -> bool:
    return "\t" in line or line.count("|") >= 2


def _split_paragraph_sections(text: str) -> list[TextSpan]:
    sections: list[TextSpan] = []
    start = 0
    for match in re.finditer(r"\n\s*\n+", text):
        end = match.start()
        trimmed_start, trimmed_end = _trim_span(text, start, end)
        if trimmed_start < trimmed_end:
            sections.append(TextSpan(text[trimmed_start:trimmed_end], trimmed_start, trimmed_end))
        start = match.end()

    trimmed_start, trimmed_end = _trim_span(text, start, len(text))
    if trimmed_start < trimmed_end:
        sections.append(TextSpan(text[trimmed_start:trimmed_end], trimmed_start, trimmed_end))
    return sections


def _chunk_window(
    text: str,
    window_start: int,
    window_end: int,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextSpan]:
    """Split a single text window using character overlap."""

    spans: list[TextSpan] = []
    start = _first_non_space(text, window_start)
    while start < window_end:
        raw_end = min(start + chunk_size, window_end)
        end = raw_end if raw_end == window_end else _choose_breakpoint(text, start, raw_end)
        trimmed_start, trimmed_end = _trim_span(text, start, end)

        if trimmed_start < trimmed_end:
            spans.append(TextSpan(text[trimmed_start:trimmed_end], trimmed_start, trimmed_end))

        if end >= window_end:
            break

        next_start = max(end - chunk_overlap, start + 1)
        start = _first_non_space(text, next_start)

    return spans


def _previous_nonblank_index(line_spans: list[tuple[int, int, str]], index: int) -> int | None:
    for candidate in range(index - 1, -1, -1):
        if line_spans[candidate][2].strip():
            return candidate
    return None


def _next_nonblank_index(line_spans: list[tuple[int, int, str]], index: int) -> int | None:
    for candidate in range(index + 1, len(line_spans)):
        if line_spans[candidate][2].strip():
            return candidate
    return None


def _choose_breakpoint(text: str, start: int, raw_end: int) -> int:
    min_end = start + max((raw_end - start) // 2, 1)
    best = -1
    best_width = 0
    for token in BREAKPOINTS:
        position = text.rfind(token, min_end, raw_end)
        if position > best:
            best = position
            best_width = len(token)

    if best == -1:
        return raw_end
    return best + best_width


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _first_non_space(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    return start


def _chunk_id(document_id: str, start_char: int, end_char: int) -> str:
    payload = f"{document_id}:{start_char}:{end_char}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
