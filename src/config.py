"""Central configuration for Local Novel AI v3 (LM Studio Edition)."""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_NOVELS_DIR = os.path.join(ROOT_DIR, "data", "novels")   # optional corpus
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")                 # project.json, chapters, epub
CHAPTERS_DIR = os.path.join(OUTPUT_DIR, "chapters")
RAG_DB_DIR = os.path.join(ROOT_DIR, "rag_db")                 # ChromaDB persistent store
PROJECT_JSON = os.path.join(OUTPUT_DIR, "project.json")
SQLITE_PATH = os.path.join(OUTPUT_DIR, "novel_memory.sqlite3")

for _d in (DATA_NOVELS_DIR, OUTPUT_DIR, CHAPTERS_DIR, RAG_DB_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# LM Studio (OpenAI-compatible server)
# ---------------------------------------------------------------------------
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:12340/v1")
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")  # any string works locally
DEFAULT_MODEL = os.environ.get("LMSTUDIO_MODEL", "local-model")

# Generation defaults
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "0.85"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "0.95"))
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "4096"))
REVIEW_TEMPERATURE = float(os.environ.get("REVIEW_TEMPERATURE", "0.3"))

# ---------------------------------------------------------------------------
# Novel pipeline defaults
# ---------------------------------------------------------------------------
TARGET_CHARS = int(os.environ.get("TARGET_CHARS", "100000"))     # 100k-character target
CHAPTER_MIN_CHARS = int(os.environ.get("CHAPTER_MIN_CHARS", "1500"))
CHAPTER_TARGET_CHARS = int(os.environ.get("CHAPTER_TARGET_CHARS", "2500"))
MAX_REWRITE_ROUNDS = int(os.environ.get("MAX_REWRITE_ROUNDS", "2"))

# RAG chunking
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))           # characters per chunk
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
