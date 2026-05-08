"""FastAPI backend for the RAG Copilot."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .chunking import chunk_documents
from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_MIN_SCORE, DEFAULT_TOP_K, INDEX_DIR, PROJECT_ROOT, RAW_DOCS_DIR
from .ingest import SUPPORTED_SUFFIXES, load_documents
from .llm import DEFAULT_DASHSCOPE_MODEL, create_model
from .rag_chain import RAGChain
from .retriever import INDEX_FILE, VectorRetriever


def load_env_file(path: Path) -> None:
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


load_env_file(PROJECT_ROOT / ".env")


def _default_llm_provider() -> str:
    return os.getenv("RAG_LLM_PROVIDER", "dashscope")


def _cors_origins() -> list[str]:
    raw = os.getenv("RAG_CORS_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


app = FastAPI(title="RAG Copilot API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever_cache: VectorRetriever | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    min_score: float = Field(DEFAULT_MIN_SCORE, ge=0)
    llm: Literal["mock", "dashscope"] | None = None
    model: str | None = None
    temperature: float = Field(0.2, ge=0, le=2)
    max_tokens: int = Field(800, ge=1, le=8192)
    mock_response: str = "这是一个 MockLLM 测试答案。[1]"
    show_prompt: bool = False


class CitationResponse(BaseModel):
    source: str
    chunk_id: str
    score: float
    start_char: int
    end_char: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    found_evidence: bool
    prompt: str | None = None


class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    index_path: str
    default_llm: str
    model: str


class DocumentResponse(BaseModel):
    name: str
    size: int
    suffix: str


class UploadResponse(BaseModel):
    files: list[DocumentResponse]


class RebuildIndexRequest(BaseModel):
    chunk_size: int = Field(DEFAULT_CHUNK_SIZE, ge=100, le=4000)
    chunk_overlap: int = Field(DEFAULT_CHUNK_OVERLAP, ge=0, le=1000)


class RebuildIndexResponse(BaseModel):
    documents: int
    chunks: int
    index_path: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index_path = INDEX_DIR / INDEX_FILE
    return HealthResponse(
        status="ok",
        index_ready=index_path.exists(),
        index_path=str(index_path),
        default_llm=_default_llm_provider(),
        model=os.getenv("DASHSCOPE_MODEL", DEFAULT_DASHSCOPE_MODEL),
    )


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents() -> list[DocumentResponse]:
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    documents: list[DocumentResponse] = []
    for path in sorted(RAW_DOCS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.append(
                DocumentResponse(
                    name=path.name,
                    size=path.stat().st_size,
                    suffix=path.suffix.lower(),
                )
            )
    return documents


@app.post("/documents/upload", response_model=UploadResponse)
def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    saved_files: list[DocumentResponse] = []

    for upload in files:
        safe_name = _safe_upload_name(upload.filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}",
            )

        target = RAW_DOCS_DIR / safe_name
        with target.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        saved_files.append(
            DocumentResponse(
                name=target.name,
                size=target.stat().st_size,
                suffix=suffix,
            )
        )

    return UploadResponse(files=saved_files)


@app.post("/index/rebuild", response_model=RebuildIndexResponse)
def rebuild_index(request: RebuildIndexRequest | None = None) -> RebuildIndexResponse:
    global _retriever_cache

    request = request or RebuildIndexRequest()
    try:
        documents = load_documents(RAW_DOCS_DIR)
        chunks = chunk_documents(
            documents,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        retriever = VectorRetriever.from_chunks(chunks)
        index_path = retriever.save(INDEX_DIR)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _retriever_cache = retriever
    return RebuildIndexResponse(
        documents=len(documents),
        chunks=len(chunks),
        index_path=str(index_path),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    retriever = _get_retriever()
    provider = request.llm or _default_llm_provider()

    try:
        llm = create_model(
            provider,
            mock_response=request.mock_response,
            model=request.model,
            base_url=None,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        chain = RAGChain(
            retriever,
            llm,
            top_k=request.top_k,
            min_score=request.min_score,
        )
        response = chain.answer(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AskResponse(
        answer=response.answer,
        citations=[
            CitationResponse(
                source=citation.source,
                chunk_id=citation.chunk_id,
                score=citation.score,
                start_char=citation.start_char,
                end_char=citation.end_char,
            )
            for citation in response.citations
        ],
        found_evidence=response.found_evidence,
        prompt=response.prompt if request.show_prompt else None,
    )


def _get_retriever() -> VectorRetriever:
    global _retriever_cache

    if _retriever_cache is not None:
        return _retriever_cache

    index_path = INDEX_DIR / INDEX_FILE
    if not index_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Index not found: {index_path}. Run scripts/build_index.py first.",
        )

    _retriever_cache = VectorRetriever.load(INDEX_DIR)
    return _retriever_cache


def _safe_upload_name(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name
