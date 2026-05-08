"""Project-level paths and defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"

DEFAULT_CHUNK_SIZE = 450
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_VECTOR_DIMENSION = 4096
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.03
