"""Gradio UI for Local Novel AI v3 — LM Studio Edition.

Run:  python src/app.py   ->  http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr  # noqa: E402

import config  # noqa: E402
from orchestrator import NovelOrchestrator  # noqa: E402
from epub_export import export_epub, load_chapters_from_db  # noqa: E402


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
ORCH = NovelOrchestrator()
LOG_LINES: list[str] = []
GEN_THREAD: threading.Thread | None = None
LOCK = threading.Lock()


def log_sink(line: str) -> None:
    with LOCK:
        LOG_LINES.append(line)
        if len(LOG_LINES) > 2000:
            del LOG_LINES[:-2000]


ORCH.log_cb = log_sink


def fmt_status() -> str:
    st = ORCH.status()
    if "error" in st:
        return f"**{st['error']}**"
    pct = min(100.0, 100.0 * st["total_chars"] / max(1, st["target_chars"]))
    return (
        f"| Field | Value |\n|---|---|\n"
        f"| Project | `{st['project_id']}` |\n"
        f"| Title | {st['title']} |\n"
        f"| Status | {st['status']} |\n"
        f"| Chapters | {st['chapters']} |\n"
        f"| Characters | {st['characters']} |\n"
        f"| World facts | {st['world_facts']} |\n"
        f"| RAG docs | {json.dumps(st['rag_docs'])} |\n\n"
        f"**Progress:** {st['total_chars']:,} / {st['target_chars']:,} characters ({pct:.1f}%)"
    )


def fmt_log() -> str:
    with LOCK:
        return "\n".join(LOG_LINES[-200:])


def running() -> bool:
    return GEN_THREAD is not None and GEN_THREAD.is_alive()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def act_test_connection(base_url: str, model: str):
    ORCH.llm.base_url = base_url or config.LMSTUDIO_BASE_URL
    ORCH.llm.client.base_url = base_url or config.LMSTUDIO_BASE_URL
    ORCH.llm.model = model or config.DEFAULT_MODEL
    res = ORCH.llm.ping()
    if res["ok"]:
        return (f"✅ Connected to {res['base_url']}. Models: "
                f"{', '.join(res['models']) or '(none listed — server may not expose /models)'}")
    return f"❌ Cannot reach LM Studio at {res['base_url']}: {res['error']}"


def act_new_project(premise, title, genre, style, target_chars, chapter_chars):
    if running():
        return "Generation is running — press Stop first.", fmt_status()
    if not premise.strip():
        return "Premise is required.", fmt_status()
    LOG_LINES.clear()
    p = ORCH.new_project(premise, title, genre, style,
                         int(target_chars), int(chapter_chars))
    ORCH.save_project()
    return (f"Created project `{p['project_id']}`. Now press **Generate Outline**, "
            f"then **Start Writing**."), fmt_status()


def act_resume():
    p = ORCH.load_project()
    if not p:
        return "No saved project found at output/project.json", fmt_status()
    return (f"Resumed `{p['project_id']}` — {p.get('status')} "
            f"({ORCH.memory.total_chars(p['project_id']):,} chars so far)."), fmt_status()


def act_outline():
    if running():
        return "Generation running — stop first.", fmt_status()
    if not ORCH.project:
        return "Create a project first.", fmt_status()
    try:
        out = ORCH.make_outline()
        beats = "\n".join(f"{b['chapter']}. [{b['arc']}] {b['beat']}"
                          for b in out["beats"][:80])
        return f"Outline generated ({len(out['beats'])} beats):\n\n{beats}", fmt_status()
    except Exception as e:
        return f"Outline failed: {e}", fmt_status()


def _bg_run(batch: int | None):
    global GEN_THREAD
    try:
        ORCH.reset_stop()
        ORCH.run(max_chapters=batch)
    except Exception as e:
        log_sink(f"[ERROR] {e}")
    finally:
        log_sink("— generation thread finished —")


def act_start(base_url, model, batch):
    global GEN_THREAD
    if running():
        return "Already running…", fmt_status(), fmt_log()
    if not ORCH.project:
        p = ORCH.load_project()
        if not p:
            return "Create or resume a project first.", fmt_status(), fmt_log()
    ORCH.llm.base_url = base_url or config.LMSTUDIO_BASE_URL
    ORCH.llm.client.base_url = ORCH.llm.base_url
    ORCH.llm.model = model or config.DEFAULT_MODEL
    n = int(batch) if batch else None
    ORCH.reset_stop()  # clear any earlier Stop so the new run isn't killed instantly
    GEN_THREAD = threading.Thread(target=_bg_run, args=(n,), daemon=True)
    GEN_THREAD.start()
    return ("Started writing loop in background (each chapter: beat → RAG → memory → "
            "generate → review → rewrite → summarize → extract → store)."), fmt_status(), fmt_log()


def act_stop():
    if running():
        ORCH.stop()
        return "Stop requested — finishing current step.", fmt_status(), fmt_log()
    return "Nothing is running.", fmt_status(), fmt_log()


def act_refresh():
    return fmt_status(), fmt_log()


def act_preview_chapter(num):
    if not ORCH.project:
        return "No project loaded."
    ch = ORCH.memory.get_chapter(ORCH.project["project_id"], int(num))
    if not ch:
        return f"Chapter {num} not found."
    return f"# Chapter {ch['number']}: {ch['title']}\n\n{ch['content']}\n\n**Summary:** {ch['summary']}"


def act_characters():
    if not ORCH.project:
        return "No project loaded."
    chars = ORCH.memory.get_characters(ORCH.project["project_id"])
    if not chars:
        return "(no characters yet)"
    rows = ["| Name | Role | Status | Location | Description | Goals | Traits |",
            "|---|---|---|---|---|---|---|"]
    for c in chars:
        traits = ", ".join(c["traits"]) if isinstance(c["traits"], list) else str(c["traits"])
        rows.append(f"| {c['name']} | {c['role']} | {c['status']} | {c['location']} "
                    f"| {c['description']} | {c['goals']} | {traits} |")
    return "\n".join(rows)


def act_facts():
    if not ORCH.project:
        return "No project loaded."
    facts = ORCH.memory.get_world_facts(ORCH.project["project_id"])
    events = ORCH.memory.recent_events(ORCH.project["project_id"], limit=40)
    fb = "\n".join(f"- [{f['category']}@Ch.{f['source_chapter']}] {f['fact']}" for f in facts)
    eb = "\n".join(f"- Ch.{e['chapter_number']}: {e['event']}" for e in events)
    return f"### World facts\n{fb or '(none)'}\n\n### Recent events\n{eb or '(none)'}"


def act_export():
    if not ORCH.project:
        return None, "No project loaded."
    pid = ORCH.project["project_id"]
    chapters = load_chapters_from_db(ORCH.memory, pid)
    if not chapters:
        return None, "No chapters written yet."
    path = export_epub(ORCH.project, chapters)
    return path, f"Exported {len(chapters)} chapters → output/{os.path.basename(path)}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Local Novel AI v3 — LM Studio") as demo:
    gr.Markdown("# 📖 Local Novel AI v3 — LM Studio Edition\n"
                "LM Studio + ChromaDB RAG + SQLite character/world memory + automatic "
                "plotting, generation, review & rewriting → EPUB.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### LM Studio connection")
            base_url = gr.Textbox(label="Base URL", value=config.LMSTUDIO_BASE_URL)
            model_id = gr.Textbox(label="Model id", value=config.DEFAULT_MODEL,
                                  info="Change if LM Studio reports a different loaded-model id.")
            test_btn = gr.Button("Test connection")
            test_out = gr.Textbox(label="Connection status", interactive=False)
        with gr.Column(scale=2):
            gr.Markdown("### New project")
            premise = gr.Textbox(label="Premise", lines=3,
                                 placeholder="A lighthouse keeper discovers the light answers back…")
            title = gr.Textbox(label="Title (optional)")
            genre = gr.Textbox(label="Genre (optional)", placeholder="sci-fi horror")
            style = gr.Textbox(label="Style notes (optional)",
                               placeholder="lyrical, close third person, unreliable narrator")
            target_chars = gr.Number(label="Target characters", value=config.TARGET_CHARS, precision=0)
            chapter_chars = gr.Number(label="Chars per chapter", value=config.CHAPTER_TARGET_CHARS, precision=0)
            new_btn = gr.Button("Create project", variant="primary")
            resume_btn = gr.Button("Resume saved project")
    proj_msg = gr.Textbox(label="Project message", interactive=False)

    with gr.Row():
        outline_btn = gr.Button("1️⃣ Generate Outline")
        start_btn = gr.Button("2️⃣ Start Writing", variant="primary")
        batch = gr.Number(label="Chapters per batch (0 = until target)", value=0, precision=0)
        stop_btn = gr.Button("⏹ Stop", variant="stop")

    outline_out = gr.Textbox(label="Outline", lines=10, interactive=False)

    with gr.Row():
        status_md = gr.Markdown(fmt_status())
    refresh_btn = gr.Button("Refresh status / log")
    log_out = gr.Textbox(label="Pipeline log", lines=14, interactive=False)

    with gr.Tabs():
        with gr.Tab("Chapter preview"):
            chap_num = gr.Number(label="Chapter number", value=1, precision=0)
            preview_btn = gr.Button("Load chapter")
            preview_out = gr.Markdown()
        with gr.Tab("Characters (SQLite)"):
            char_btn = gr.Button("Reload characters")
            char_out = gr.Markdown()
        with gr.Tab("World & events"):
            fact_btn = gr.Button("Reload memory")
            fact_out = gr.Markdown()
        with gr.Tab("EPUB export"):
            export_btn = gr.Button("Export EPUB")
            export_msg = gr.Textbox(label="Export message", interactive=False)
            file_out = gr.File(label="Download EPUB")

    # auto-refresh every few seconds while running
    timer = gr.Timer(5.0)
    timer.tick(fn=lambda: (fmt_status(), fmt_log()), outputs=[status_md, log_out])

    # wiring
    test_btn.click(act_test_connection, [base_url, model_id], test_out)
    new_btn.click(act_new_project,
                  [premise, title, genre, style, target_chars, chapter_chars],
                  [proj_msg, status_md])
    resume_btn.click(act_resume, None, [proj_msg, status_md])
    outline_btn.click(act_outline, None, [outline_out, status_md])
    start_btn.click(act_start, [base_url, model_id, batch], [proj_msg, status_md, log_out])
    stop_btn.click(act_stop, None, [proj_msg, status_md, log_out])
    refresh_btn.click(act_refresh, None, [status_md, log_out])
    preview_btn.click(act_preview_chapter, chap_num, preview_out)
    char_btn.click(act_characters, None, char_out)
    fact_btn.click(act_facts, None, fact_out)
    export_btn.click(act_export, None, [file_out, export_msg])


if __name__ == "__main__":
    # Try to resume an existing project automatically.
    try:
        if ORCH.load_project():
            log_sink(f"Auto-resumed project '{ORCH.project['project_id']}'")
    except Exception as e:  # pragma: no cover
        print("No project to resume:", e)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
