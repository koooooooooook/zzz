"""Chapter generation: prompts assembled from plot beat + RAG + SQLite memory."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from llm import LMStudioClient
import config


WRITER_SYSTEM = """You are a versatile novelist writing long-form fiction in polished prose.
Hard rules:
- Show, don't tell; vary sentence rhythm; keep characters consistent with the provided memory.
- Never contradict the established character/world memory or prior chapter summaries.
- End every chapter on a hook (question, reversal, threat, or decision).
- Output ONLY the chapter itself, formatted as:
# <Title>
<prose paragraphs>
Do not add author notes, summaries, or meta commentary."""


def _format_characters(chars: List[Dict[str, Any]]) -> str:
    if not chars:
        return "(no characters registered yet)"
    lines = []
    for c in chars:
        traits = ", ".join(c.get("traits") or []) if isinstance(c.get("traits"), list) else str(c.get("traits", ""))
        lines.append(
            f"- {c['name']} ({c.get('role') or '?'}, {c.get('status') or 'alive'}): "
            f"{c.get('description','')} | traits: {traits} | location: {c.get('location','')} | goals: {c.get('goals','')}"
        )
    return "\n".join(lines)


def _format_world(facts: List[Dict[str, Any]]) -> str:
    if not facts:
        return "(no world facts registered yet)"
    return "\n".join(f"- [{f['category']}] {f['fact']}" for f in facts[-40:])


def _format_summaries(sums: List[Dict[str, Any]], last_n: int = 6) -> str:
    if not sums:
        return "(this is the first chapter)"
    tail = sums[-last_n:]
    return "\n".join(f"Ch.{s['number']} {s['title']}: {s['summary']}" for s in tail)


def build_writer_prompt(project: Dict[str, Any], chapter_number: int, beat: str,
                        rag_docs: List[str], characters: List[Dict[str, Any]],
                        world_facts: List[Dict[str, Any]], events: List[Dict[str, Any]],
                        summaries: List[Dict[str, Any]]) -> str:
    rag_block = "\n\n".join(rag_docs) if rag_docs else "(no retrieved material)"
    event_block = "\n".join(f"- Ch.{e['chapter_number']}: {e['event']}" for e in events[-12:]) or "(none yet)"
    return f"""Novel: "{project['title']}" — {project.get('logline','')}
Genre: {project.get('genre','')} | Style notes: {project.get('style','')}

CHARACTER MEMORY (SQLite):
{_format_characters(characters)}

WORLD MEMORY (SQLite):
{_format_world(world_facts)}

RECENT EVENTS (SQLite):
{event_block}

PRIOR CHAPTER SUMMARIES:
{_format_summaries(summaries)}

RETRIEVED CONTEXT (RAG):
{rag_block}

TASK: Write Chapter {chapter_number}.
Plot beat to accomplish: {beat}
Target length: about {config.CHAPTER_TARGET_CHARS} characters of prose (never fewer than {config.CHAPTER_MIN_CHARS}).
Continue smoothly from the previous chapter's ending."""


def generate_chapter(llm: LMStudioClient, prompt: str) -> Dict[str, str]:
    raw = llm.complete(WRITER_SYSTEM, prompt,
                       temperature=config.GEN_TEMPERATURE,
                       max_tokens=config.GEN_MAX_TOKENS)
    return parse_chapter(raw, fallback_title="Untitled Chapter")


def parse_chapter(raw: str, fallback_title: str = "Untitled") -> Dict[str, str]:
    raw = (raw or "").strip()
    title = fallback_title
    body = raw
    m = re.match(r"^\s*#{1,6}\s*(.+?)\s*$", raw)
    if m:
        title = m.group(1).strip()
        body = raw[m.end():].strip()
    else:
        # Sometimes models emit "Chapter 3: Title" on the first line.
        m2 = re.match(r"^(?:Chapter\s+\d+[:\-–]\s*)?(.{0,80})\n{2,}", raw)
        if m2 and len(m2.group(1)) < 80 and not m2.group(1).endswith("."):
            title = m2.group(1).strip(":.- ") or fallback_title
            body = raw[m2.end():].strip()
    title = re.sub(r"^#{1,6}\s*", "", (title or "")).strip()
    return {"title": title or fallback_title, "content": body or raw}


SUMMARY_SYSTEM = ("You summarize novel chapters. Reply with ONE paragraph (max 120 words) "
                  "capturing concrete plot changes: who did what, where, consequences.")


def summarize_chapter(llm: LMStudioClient, chapter_text: str) -> str:
    return llm.complete(SUMMARY_SYSTEM, chapter_text[:12000],
                        temperature=0.3, max_tokens=400)


EXTRACT_SYSTEM = """You extract story state from a novel chapter into JSON.
Reply ONLY with JSON:
{
  "characters": [{"name":"...","role":"...","description":"...","traits":["..."],
                   "status":"alive|dead|unknown","location":"...","goals":"..."}],
  "world_facts": [{"category":"location|rule|item|history|general","fact":"..."}],
  "events": ["short declarative sentence of a plot event that happened"]
}
Include only information actually shown or stated in the chapter. Update existing
characters rather than inventing duplicates."""


def extract_memories(llm: LMStudioClient, chapter_text: str,
                     known_characters: List[str]) -> Dict[str, Any]:
    user = ("Known characters so far: " + (", ".join(known_characters) or "none") +
            "\n\nChapter text:\n" + chapter_text[:12000])
    raw = llm.complete(EXTRACT_SYSTEM, user, temperature=0.2, max_tokens=1500)
    data = LMStudioClient.extract_json(raw)
    if not isinstance(data, dict):
        return {"characters": [], "world_facts": [], "events": []}
    return {
        "characters": data.get("characters", []) or [],
        "world_facts": data.get("world_facts", []) or [],
        "events": data.get("events", []) or [],
    }
