"""Load raw single-topic knowledge files into normalized documents."""

from __future__ import annotations

import csv
import hashlib
import importlib
from pathlib import Path
from typing import Iterable

from .config import RAW_DOCS_DIR
from .models import Document


TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
FAQ_SUFFIXES = {".csv"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | FAQ_SUFFIXES | PDF_SUFFIXES


def load_documents(path: str | Path = RAW_DOCS_DIR, *, recursive: bool = True) -> list[Document]:
    """Load all supported files from a file or directory.

    Markdown and plain text work without third-party dependencies.
    CSV FAQ files are supported when they contain question/answer-like columns.
    PDF loading is optional and requires installing the project's `pdf` extra.
    """

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Document path does not exist: {source_path}")

    files = [source_path] if source_path.is_file() else list(_iter_source_files(source_path, recursive))
    documents: list[Document] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            documents.append(_load_text_file(file_path, source_path))
        elif suffix in FAQ_SUFFIXES:
            documents.extend(_load_faq_csv(file_path, source_path))
        elif suffix in PDF_SUFFIXES:
            documents.extend(_load_pdf(file_path, source_path))
    return [doc for doc in documents if doc.text.strip()]


def _iter_source_files(root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _load_text_file(file_path: Path, root: Path) -> Document:
    text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    source = _source_name(file_path, root)
    return Document(
        id=_stable_id(source, text),
        text=text,
        source=source,
        metadata={
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower().lstrip("."),
            "title": _infer_title(text, file_path),
        },
    )


def _load_faq_csv(file_path: Path, root: Path) -> list[Document]:
    source = _source_name(file_path, root)
    with file_path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []

        question_key = _find_column(reader.fieldnames, {"question", "q", "title", "prompt"})
        answer_key = _find_column(reader.fieldnames, {"answer", "a", "content", "response"})
        if question_key is None or answer_key is None:
            raise ValueError(
                f"FAQ CSV must contain question/answer columns. Found: {', '.join(reader.fieldnames)}"
            )

        documents: list[Document] = []
        for row_number, row in enumerate(reader, start=2):
            question = (row.get(question_key) or "").strip()
            answer = (row.get(answer_key) or "").strip()
            if not question and not answer:
                continue

            text = f"Question: {question}\nAnswer: {answer}".strip()
            row_source = f"{source}#row={row_number}"
            documents.append(
                Document(
                    id=_stable_id(row_source, text),
                    text=text,
                    source=row_source,
                    metadata={
                        "file_name": file_path.name,
                        "file_type": "csv",
                        "row": row_number,
                        "title": question or file_path.stem,
                    },
                )
            )
        return documents


def _load_pdf(file_path: Path, root: Path) -> list[Document]:
    try:
        pypdf = importlib.import_module("pypdf")
    except ModuleNotFoundError as exc:
        raise RuntimeError("PDF ingestion requires installing the optional dependency: pip install .[pdf]") from exc

    source = _source_name(file_path, root)
    reader = pypdf.PdfReader(str(file_path))
    documents: list[Document] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        page_source = f"{source}#page={page_index}"
        documents.append(
            Document(
                id=_stable_id(page_source, text),
                text=text,
                source=page_source,
                metadata={
                    "file_name": file_path.name,
                    "file_type": "pdf",
                    "page": page_index,
                    "title": file_path.stem,
                },
            )
        )
    return documents


def _find_column(fieldnames: list[str], candidates: set[str]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _source_name(path: Path, root: Path) -> str:
    root_dir = root if root.is_dir() else root.parent
    try:
        return path.resolve().relative_to(root_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _infer_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
        if stripped:
            return stripped[:80]
    return path.stem


def _stable_id(source: str, text: str) -> str:
    payload = f"{source}\0{text}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]
