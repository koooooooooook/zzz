"""ChromaDB vector store wrapper for the novel RAG pipeline.

Each project gets its own persistent collection under rag_db/<project_id>/.
Collections:
  - reference: chunks of user-supplied corpus TXT files (style/reference material)
  - chapters:  summaries + full chapter text of the generated novel
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

import config


def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character chunks on paragraph boundaries."""
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= size:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            # hard-split very long paragraphs
            while len(p) > size:
                head = p[:size]
                tail_start = max(0, size - overlap)
                chunks.append(head)
                p = p[tail_start:]
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


class NovelRAG:
    def __init__(self, project_id: str):
        self.project_id = project_id
        persist_dir = os.path.join(config.RAG_DB_DIR, project_id)
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False))

    # -- collections ----------------------------------------------------------
    def _coll(self, name: str):
        return self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"})

    # -- reference corpus --------------------------------------------------------
    def add_reference(self, doc_name: str, text: str) -> int:
        coll = self._coll("reference")
        chunks = chunk_text(text)
        n = 0
        for i, c in enumerate(chunks):
            coll.upsert(ids=[f"{doc_name}::{i}"], documents=[c],
                        metadatas=[{"source": doc_name, "chunk": i}])
            n += 1
        return n

    def query_reference(self, qtext: str, k: int = config.RAG_TOP_K) -> List[str]:
        coll = self._coll("reference")
        if coll.count() == 0:
            return []
        res = coll.query(query_texts=[qtext], n_results=min(k, coll.count()))
        return list(res.get("documents", [[]])[0])

    # -- chapter memory ------------------------------------------------------------
    def add_chapter(self, number: int, title: str, summary: str, content: str) -> None:
        coll = self._coll("chapters")
        coll.upsert(ids=[f"chapter::{number}::summary"],
                    documents=[f"Chapter {number} — {title}\nSummary: {summary}"],
                    metadatas=[{"chapter": number, "kind": "summary", "title": title}])
        for i, c in enumerate(chunk_text(content)):
            coll.upsert(ids=[f"chapter::{number}::chunk::{i}"],
                        documents=[c],
                        metadatas=[{"chapter": number, "kind": "text", "title": title}])

    def query_chapters(self, qtext: str, k: int = config.RAG_TOP_K) -> List[str]:
        coll = self._coll("chapters")
        if coll.count() == 0:
            return []
        res = coll.query(query_texts=[qtext], n_results=min(k, coll.count()))
        docs = list(res.get("documents", [[]])[0])
        metas = list(res.get("metadatas", [[]])[0])
        out = []
        for d, m in zip(docs, metas):
            ch = (m or {}).get("chapter", "?")
            kind = (m or {}).get("kind", "")
            out.append(f"[Chapter {ch} | {kind}] {d}")
        return out

    def counts(self) -> Dict[str, int]:
        return {"reference": self._coll("reference").count(),
                "chapters": self._coll("chapters").count()}
