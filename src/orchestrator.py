"""Novel Orchestrator: end-to-end pipeline state machine.

Each chapter runs:
1. select next plot beat  2. RAG retrieval  3. load SQLite memory
4. generate chapter       5. AI review      6. auto-rewrite if failed
7. summarize              8. extract memories
9. persist to SQLite + ChromaDB + output/chapters/*.md
10. loop until target character count reached.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import config
from llm import LMStudioClient
from memory import MemoryDB
from rag import NovelRAG
import plotter
import generator
import reviewer


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "novel").lower()).strip("-")
    return s or f"novel-{int(time.time())}"


class NovelOrchestrator:
    def __init__(self, base_url: str = config.LMSTUDIO_BASE_URL,
                 model: str = config.DEFAULT_MODEL):
        self.llm = LMStudioClient(base_url=base_url, model=model)
        self.memory = MemoryDB()
        self.project: Optional[Dict[str, Any]] = None
        self.rag: Optional[NovelRAG] = None
        self.log_cb: Optional[Callable[[str], None]] = None
        self._stop = threading.Event()

    # -- logging -------------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if self.log_cb:
            try:
                self.log_cb(line)
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()

    def reset_stop(self) -> None:
        """Clear a previous stop request (call before starting a new run)."""
        self._stop.clear()

    # -- persistence -----------------------------------------------------------
    def save_project(self) -> None:
        if self.project:
            with open(config.PROJECT_JSON, "w", encoding="utf-8") as f:
                json.dump(self.project, f, ensure_ascii=False, indent=2)

    def load_project(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(config.PROJECT_JSON):
            with open(config.PROJECT_JSON, "r", encoding="utf-8") as f:
                self.project = json.load(f)
            self.rag = NovelRAG(self.project["project_id"])
            return self.project
        return None

    # -- project lifecycle -------------------------------------------------------
    def new_project(self, premise: str, title: str = "", genre: str = "",
                    style: str = "", target_chars: int = config.TARGET_CHARS,
                    chapter_chars: int = config.CHAPTER_TARGET_CHARS) -> Dict[str, Any]:
        pid = slugify(title or premise)[:40]
        self.project = {
            "project_id": pid,
            "title": title.strip() or (premise.strip().split(".")[0][:60] or "Untitled"),
            "premise": premise.strip(),
            "genre": genre.strip(),
            "style": style.strip(),
            "target_chars": int(target_chars),
            "chapter_target_chars": int(chapter_chars),
            "logline": "",
            "outline": None,
            "status": "draft",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.rag = NovelRAG(pid)
        self.log(f"Created project '{pid}'")
        return self.project

    def make_outline(self) -> Dict[str, Any]:
        assert self.project, "no project loaded"
        self.log("Generating plot outline…")
        out = plotter.generate_outline(
            self.llm, self.project["premise"], self.project["genre"],
            self.project["target_chars"], self.project["chapter_target_chars"])
        self.project["outline"] = out
        self.project["logline"] = out.get("logline", "")
        if not self.project.get("title_from_user"):
            self.project["title"] = out.get("title") or self.project["title"]
        self.project["status"] = "outlined"
        self.save_project()
        self.log(f"Outline ready: {len(out['beats'])} beats — “{out.get('title','')}”")
        return out

    # -- one chapter --------------------------------------------------------------
    def write_chapter(self, number: int) -> Optional[Dict[str, Any]]:
        assert self.project and self.rag
        p = self.project
        beat = plotter.next_beat(p.get("outline") or {"beats": []}, number)

        # 2/3. retrieve + load memory
        chars = self.memory.get_characters(p["project_id"])
        facts = self.memory.get_world_facts(p["project_id"])
        events = self.memory.recent_events(p["project_id"])
        sums = self.memory.chapter_summaries(p["project_id"])
        query = f"{p['title']} {p.get('logline','')} chapter {number} beat: {beat}\n" \
                f"last summary: {sums[-1]['summary'] if sums else 'opening of the novel'}"
        rag_docs = self.rag.query_chapters(query) + self.rag.query_reference(query)
        self.log(f"Ch.{number}: beat='{beat[:70]}…' retrieved {len(rag_docs)} RAG docs, "
                 f"{len(chars)} characters, {len(facts)} facts")

        # 4. generate
        prompt = generator.build_writer_prompt(p, number, beat, rag_docs, chars,
                                               facts, events, sums)
        ch = generator.generate_chapter(self.llm, prompt)
        self.log(f"Ch.{number}: draft {len(ch['content'])} chars — “{ch['title']}”")

        # 5/6. review -> rewrite loop
        passed = False
        for attempt in range(config.MAX_REWRITE_ROUNDS + 1):
            if self._stop.is_set():
                return None
            rev = reviewer.review_chapter(self.llm, p, number, beat, ch["content"],
                                          chars, facts, sums)
            self.log(f"Ch.{number}: review score={rev['score']} pass={rev['pass']} "
                     f"(attempt {attempt + 1})")
            if rev["pass"]:
                passed = True
                break
            if attempt < config.MAX_REWRITE_ROUNDS:
                if len(ch["content"]) < config.CHAPTER_MIN_CHARS:
                    # too short / stub — regenerate from scratch instead of rewriting
                    ch = generator.generate_chapter(self.llm, prompt)
                else:
                    ch = reviewer.rewrite_chapter(self.llm, ch["title"], ch["content"],
                                                  rev, beat)
                self.log(f"Ch.{number}: rewrote → {len(ch['content'])} chars")

        # 7. summarize
        summary = generator.summarize_chapter(self.llm, ch["content"])

        # 8. extract memories
        mem = generator.extract_memories(self.llm, ch["content"],
                                         [c["name"] for c in chars])
        for c in mem["characters"]:
            self.memory.upsert_character(p["project_id"], c.get("name", ""), **{
                k: v for k, v in c.items() if k != "name"})
        for wf in mem["world_facts"]:
            self.memory.add_world_fact(p["project_id"], wf.get("fact", ""),
                                       wf.get("category", "general"), number)
        for ev in mem["events"]:
            self.memory.add_event(p["project_id"], number, str(ev))

        # 9. store everywhere
        self.memory.save_chapter(p["project_id"], number, ch["title"],
                                 ch["content"], summary, passed)
        self.rag.add_chapter(number, ch["title"], summary, ch["content"])
        chap_path = os.path.join(config.CHAPTERS_DIR, f"chapter_{number:03d}.md")
        with open(chap_path, "w", encoding="utf-8") as f:
            f.write(f"# Chapter {number}: {ch['title']}\n\n{ch['content']}\n\n---\nSummary: {summary}\n")
        p["status"] = "writing"
        p["total_chars"] = self.memory.total_chars(p["project_id"])
        self.save_project()
        self.log(f"Ch.{number}: stored ({p['total_chars']}/{p['target_chars']} chars)")
        return {"number": number, "title": ch["title"], "chars": len(ch["content"]),
                "passed": passed, "summary": summary}

    # -- run loop --------------------------------------------------------------------
    def run(self, max_chapters: Optional[int] = None) -> str:
        """Generate chapters until target reached / stopped / max_chapters."""
        assert self.project and self.rag
        p = self.project
        # NOTE: do NOT clear a stop that was set before run() started.
        if not p.get("outline"):
            self.make_outline()
        final = "stopped"
        while not self._stop.is_set():
            n = self.memory.last_chapter_number(p["project_id"]) + 1
            total = self.memory.total_chars(p["project_id"])
            if total >= p["target_chars"]:
                self.log(f"Target reached: {total}/{p['target_chars']} chars after {n - 1} chapters.")
                final = "complete"
                break
            result = self.write_chapter(n)
            if result is None:
                final = "stopped"
                break
            if max_chapters is not None:
                max_chapters -= 1
                if max_chapters <= 0:
                    self.log("Batch limit reached; pausing (resume any time).")
                    final = "batch_done"
                    break
        p["status"] = "complete" if final == "complete" else "in_progress"
        self.save_project()
        return final

    # -- status ------------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        if not self.project:
            return {"error": "No project. Create or resume one."}
        pid = self.project["project_id"]
        counts = self.rag.counts() if self.rag else {}
        return {
            "project_id": pid,
            "title": self.project["title"],
            "status": self.project.get("status"),
            "chapters": self.memory.last_chapter_number(pid),
            "total_chars": self.memory.total_chars(pid),
            "target_chars": self.project["target_chars"],
            "characters": len(self.memory.get_characters(pid)),
            "world_facts": len(self.memory.get_world_facts(pid)),
            "rag_docs": counts,
        }
