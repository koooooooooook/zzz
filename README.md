# Local Novel AI v3 — LM Studio Edition

LM Studio + ChromaDB + SQLite character/world memory + automatic plotting +
chapter generation + AI review + automatic rewriting + 100k-character target +
EPUB + Gradio.

## Architecture

```
Gradio (src/app.py)
  -> Novel Orchestrator (src/orchestrator.py)
      -> LM Studio OpenAI-compatible API (src/llm.py)
      -> ChromaDB RAG (src/rag.py)
      -> SQLite structured memory (src/memory.py)
      -> Plot (src/plotter.py) / Chapter (src/generator.py) /
         Review+Rewriter (src/reviewer.py) pipeline
      -> EPUB (src/epub_export.py)
```

## Requirements

- Windows 10/11
- Python 3.10+
- LM Studio
- A local instruct/chat model
- NVIDIA GPU is recommended, but LM Studio can also use CPU/other supported backends.

## 1. Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. LM Studio

Load a suitable local instruct model in LM Studio.

Start the local server from LM Studio's Developer/Server UI, or use the
LM Studio CLI if installed.

The default endpoint in this project is:

    http://127.0.0.1:12340/v1

The default model is `local-model`. Change it in the UI if LM Studio reports
a different loaded model identifier.

Both values are editable in the Gradio UI ("LM Studio connection") and can be
overridden with the environment variables `LMSTUDIO_BASE_URL` and `LMSTUDIO_MODEL`.

## 3. Optional training corpus

Put TXT files you have permission to use in (subfolders are supported — they are scanned recursively):

    data/novels/

The project does NOT require fine-tuning. Those files can instead be imported
into ChromaDB as reference material.

Run:

```powershell
python src/ingest.py
```

This creates searchable RAG chunks (pass an optional project id argument to
scope the corpus to one project, e.g. `python src/ingest.py my-novel`).

## 4. Run

```powershell
python src/app.py
```

Open:

    http://127.0.0.1:7860

Workflow in the UI: **Test connection** → create a project (premise, genre,
style, targets) → **Generate Outline** → **Start Writing** (optionally limit
chapters per batch) → watch the live log → preview chapters/characters/world
memory → **Export EPUB**. Use **Stop** anytime; press **Resume saved project**
(or just relaunch) to continue where you left off.

## 5. Generation pipeline

Each chapter runs:

1. Select the next plot beat.
2. Retrieve relevant prior material from ChromaDB.
3. Load structured character/world memory.
4. Generate chapter.
5. Ask LM Studio to review the chapter.
6. If review fails, automatically rewrite it (up to `MAX_REWRITE_ROUNDS`;
   stub/too-short drafts are regenerated instead of rewritten).
7. Summarize the final chapter.
8. Extract character/world/event memory.
9. Store chapter + summary + memories in ChromaDB/SQLite.
10. Continue until the target character count is reached.

## 6. Resume

The current project is saved to:

    output/project.json

SQLite memory:

    output/novel_memory.sqlite3

ChromaDB:

    rag_db/<project_id>/

Chapters:

    output/chapters/

## 7. EPUB

The UI can export all completed chapters into:

    output/<project_id>.epub

## Configuration

All defaults live in `src/config.py` and can be overridden via environment
variables (`TARGET_CHARS`, `CHAPTER_TARGET_CHARS`, `GEN_TEMPERATURE`,
`MAX_REWRITE_ROUNDS`, `RAG_TOP_K`, …). The 100,000-character target is the
default stopping condition.

## Important

A 100,000-character target is a stopping condition, not a guarantee of literary
quality. The review/rewrite loop improves continuity but should still be
human-reviewed before publication.

Use only source material you are authorized to use.
