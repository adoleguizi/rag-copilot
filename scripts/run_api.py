"""Run the FastAPI backend without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "rag_copilot.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        app_dir=str(PROJECT_ROOT / "src"),
    )
