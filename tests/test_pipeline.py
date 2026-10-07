"""Offline end-to-end test of the Local Novel AI v3 pipeline.

Runs a mock LM Studio server, then exercises:
  ingest -> outline -> chapter w/ review-fail + auto-rewrite -> memory extraction
  -> SQLite persistence -> RAG retrieval -> resume from disk -> EPUB export.

Usage:  python tests/test_pipeline.py     (exit 0 = all checks passed)
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")

# --- isolate state into a temp dir BEFORE importing config -------------------
TMP = tempfile.mkdtemp(prefix="noveltest_")
os.environ["OUTPUT_DIR_OVERRIDE"] = TMP  # (not read by config; we patch below)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
config.ROOT_DIR = TMP
config.DATA_NOVELS_DIR = os.path.join(TMP, "data", "novels")
config.OUTPUT_DIR = os.path.join(TMP, "output")
config.CHAPTERS_DIR = os.path.join(config.OUTPUT_DIR, "chapters")
config.RAG_DB_DIR = os.path.join(TMP, "rag_db")
config.PROJECT_JSON = os.path.join(config.OUTPUT_DIR, "project.json")
config.SQLITE_PATH = os.path.join(config.OUTPUT_DIR, "novel_memory.sqlite3")
for d in (config.DATA_NOVELS_DIR, config.CHAPTERS_DIR, config.RAG_DB_DIR):
    os.makedirs(d, exist_ok=True)
# small target so the loop terminates quickly
config.TARGET_CHARS = 6000
config.CHAPTER_TARGET_CHARS = 2500

import mock_lmstudio  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    srv = mock_lmstudio.serve(port=18777)
    base_url = "http://127.0.0.1:18777/v1"

    # ---- 1. optional corpus ingest ------------------------------------------
    from rag import NovelRAG, chunk_text
    corpus = ("The gulls argued with the fog all morning. Kael ignored them.\n\n"
              "By noon the beacon had begun to stutter in a rhythm too regular for machinery.")
    doc = os.path.join(config.DATA_NOVELS_DIR, "sample.txt")
    with open(doc, "w", encoding="utf-8") as f:
        f.write(corpus * 40)

    # run ingest.py as CLI against a scratch project id
    sys.argv = ["ingest.py", "scratch-corpus"]
    import ingest
    ingest.main()
    scratch_rag = NovelRAG("scratch-corpus")
    check("ingest creates reference chunks", scratch_rag.counts()["reference"] > 0,
          str(scratch_rag.counts()))
    hits = scratch_rag.query_reference("beacon stuttering rhythm")
    check("reference query returns docs", len(hits) > 0)

    # ---- 2. orchestrator: new project + outline ------------------------------
    from orchestrator import NovelOrchestrator
    orch = NovelOrchestrator(base_url=base_url, model="local-model")
    logs: list[str] = []
    orch.log_cb = lambda s: logs.append(s)

    ping = orch.llm.ping()
    check("LLM ping OK", ping.get("ok") is True, str(ping))
    check("model listed", "local-model" in ping.get("models", []))

    proj = orch.new_project(
        premise="A lighthouse keeper discovers the light answers back.",
        title="", genre="sci-fi horror", style="lyrical, close third person",
        target_chars=config.TARGET_CHARS, chapter_chars=config.CHAPTER_TARGET_CHARS)
    check("project created", bool(proj["project_id"]) and proj["status"] == "draft")

    out = orch.make_outline()
    check("outline generated", len(out.get("beats", [])) >= 20, str(len(out.get("beats", []))))
    check("outline persisted", os.path.exists(config.PROJECT_JSON))

    # ---- 3. run writing loop until target ------------------------------------
    final = orch.run()
    check("run reaches target", final == "complete", f"final={final}")
    total = orch.memory.total_chars(proj["project_id"])
    check("total chars >= target", total >= config.TARGET_CHARS, f"{total}")
    n_ch = orch.memory.last_chapter_number(proj["project_id"])
    check("multiple chapters written", n_ch >= 2, f"chapters={n_ch}")

    ch1 = orch.memory.get_chapter(proj["project_id"], 1)
    check("chapter stored in SQLite", ch1 is not None and len(ch1["content"]) > config.CHAPTER_MIN_CHARS)
    check("review passed flag set", bool(ch1["review_pass"]))
    check("summary stored", bool(ch1["summary"]))

    md_path = os.path.join(config.CHAPTERS_DIR, "chapter_001.md")
    check("chapter markdown file exists", os.path.exists(md_path))

    # rewrite path exercised? mock reviewer fails first review of each chapter
    check("auto-rewrite triggered", any("rewrote" in l for l in logs),
          "; ".join(logs[-5:]))

    chars = orch.memory.get_characters(proj["project_id"])
    facts = orch.memory.get_world_facts(proj["project_id"])
    events = orch.memory.recent_events(proj["project_id"])
    check("characters extracted", len(chars) >= 2, str(len(chars)))
    check("world facts extracted", len(facts) >= 2, str(len(facts)))
    check("events extracted", len(events) >= n_ch, f"{len(events)} vs {n_ch}")

    counts = orch.rag.counts()
    check("chapter docs in ChromaDB", counts["chapters"] >= n_ch, str(counts))
    retrieved = orch.rag.query_chapters("beacon answered sea")
    check("RAG chapter retrieval works", len(retrieved) > 0)

    # ---- 4. resume from disk (fresh process state) ----------------------------
    orch2 = NovelOrchestrator(base_url=base_url, model="local-model")
    p2 = orch2.load_project()
    check("resume loads project", p2 is not None and p2["project_id"] == proj["project_id"])
    check("resume keeps totals", orch2.memory.total_chars(p2["project_id"]) == total)
    st = orch2.status()
    check("status reports complete", st["status"] == "complete", str(st))

    # batch mode stops after requested chapters even when target not reached
    orch3 = NovelOrchestrator(base_url=base_url, model="local-model")
    p3 = orch3.new_project("Second book premise for batch test.", title="Batch Book",
                           target_chars=10 ** 9)
    orch3.make_outline()
    final3 = orch3.run(max_chapters=2)
    check("batch mode stops", final3 == "batch_done")
    check("batch wrote exactly 2", orch3.memory.last_chapter_number(p3["project_id"]) == 2,
          str(orch3.memory.last_chapter_number(p3["project_id"])))

    # stop honored: a pending Stop must prevent any further chapters
    orch3.stop()
    before = orch3.memory.last_chapter_number(p3["project_id"])
    final4 = orch3.run()
    after = orch3.memory.last_chapter_number(p3["project_id"])
    check("stop honored", final4 == "stopped" and after == before,
          f"final={final4}, chapters {before} -> {after}")
    # and after clearing the stop, a new batch resumes writing again
    orch3.reset_stop()
    final5 = orch3.run(max_chapters=1)
    check("resume after stop writes next chapter",
          final5 == "batch_done" and
          orch3.memory.last_chapter_number(p3["project_id"]) == before + 1)

    # ---- 5. EPUB export --------------------------------------------------------
    from epub_export import export_epub, load_chapters_from_db
    chapters = load_chapters_from_db(orch2.memory, proj["project_id"])
    check("epub source chapters loaded", len(chapters) == n_ch, f"{len(chapters)} != {n_ch}")
    epub_path = export_epub(orch2.project, chapters)
    check("epub file written", os.path.exists(epub_path) and os.path.getsize(epub_path) > 1000)
    import zipfile
    with zipfile.ZipFile(epub_path) as z:
        names = z.namelist()
        ok_zip = ("mimetype" in names and any(n.endswith(".opf") for n in names)
                  and any("chapter_" in n for n in names)
                  and z.testzip() is None)
        # chapters actually contain prose
        ch_files = [n for n in names if "chapter_" in n and n.endswith(".xhtml")]
        body_ok = False
        if ch_files:
            html = z.read(ch_files[0]).decode("utf-8", "replace")
            body_ok = "Kael" in html and "<p>" in html
    check("epub is a valid zip with chapters", ok_zip, str(names[:8]))
    check("epub chapter contains prose", body_ok)

    # ---- 6. DB sanity via raw sqlite -------------------------------------------
    con = sqlite3.connect(config.SQLITE_PATH)
    rows = con.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
    con.close()
    check("sqlite chapters table populated", rows >= n_ch, str(rows))

    # ---- cleanup -----------------------------------------------------------------
    srv.shutdown()
    shutil.rmtree(TMP, ignore_errors=True)

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
