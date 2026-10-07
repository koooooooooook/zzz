"""Import TXT files from data/novels/ into ChromaDB as reference RAG chunks.

Usage:
    python src/ingest.py [project_id]

The project does NOT require fine-tuning; these files become searchable
style/reference material retrieved during chapter generation. Use only
source material you are authorized to use.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from rag import NovelRAG  # noqa: E402


def read_txt(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def main() -> None:
    project_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    rag = NovelRAG(project_id)
    files = [f for f in os.listdir(config.DATA_NOVELS_DIR)
             if f.lower().endswith(".txt")]
    if not files:
        print(f"No .txt files found in {config.DATA_NOVELS_DIR}")
        print("(Optional — the pipeline works without a reference corpus.)")
        return
    total = 0
    for name in sorted(files):
        path = os.path.join(config.DATA_NOVELS_DIR, name)
        text = read_txt(path)
        n = rag.add_reference(os.path.splitext(name)[0], text)
        total += n
        print(f"  {name}: {len(text)} chars -> {n} chunks")
    print(f"Ingested {total} chunks into ChromaDB collection 'reference' "
          f"(rag_db/{project_id}/)")
    print(f"Counts: {rag.counts()}")


if __name__ == "__main__":
    main()
