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
    # Walk recursively: subfolders under data/novels/ are supported.
    # The relative path (e.g. "scifi/mars.txt") is used as the document name.
    files = []
    for dirpath, _dirnames, filenames in os.walk(config.DATA_NOVELS_DIR):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, config.DATA_NOVELS_DIR)
                files.append((rel, full))
    if not files:
        print(f"No .txt files found in {config.DATA_NOVELS_DIR}")
        print("(Optional — the pipeline works without a reference corpus.)")
        return
    total = 0
    for rel, path in sorted(files):
        text = read_txt(path)
        doc_name = os.path.splitext(rel)[0].replace(os.sep, "/")
        n = rag.add_reference(doc_name, text)
        total += n
        print(f"  {doc_name}: {len(text)} chars -> {n} chunks")
    print(f"Ingested {total} chunks into ChromaDB collection 'reference' "
          f"(rag_db/{project_id}/)")
    print(f"Counts: {rag.counts()}")


if __name__ == "__main__":
    main()
