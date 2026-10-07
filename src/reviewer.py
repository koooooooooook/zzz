"""AI review + automatic rewriting of chapters."""
from __future__ import annotations

from typing import Any, Dict, List

from llm import LMStudioClient
import config


REVIEWER_SYSTEM = """You are a rigorous continuity and craft editor for serialized fiction.
Review the chapter against the provided memory and beat. Check:
1. Continuity with character/world memory (names, status, location, established facts)
2. The assigned plot beat was actually accomplished
3. Internal logic, pacing, prose quality; no meta text or author notes in the chapter
4. Length is plausible for one chapter (not stub/fragment)
Reply ONLY with JSON:
{
  "pass": true|false,
  "score": 0-10,
  "issues": ["specific actionable problem", ...],
  "notes": "one short paragraph"
}
Set pass=false if ANY continuity break or unaccomplished beat exists."""


def review_chapter(llm: LMStudioClient, project: Dict[str, Any], chapter_number: int,
                   beat: str, chapter_text: str, characters: List[Dict[str, Any]],
                   world_facts: List[Dict[str, Any]],
                   summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    char_block = "\n".join(f"- {c['name']} ({c.get('status')}): {c.get('description','')} @ {c.get('location','')}"
                           for c in characters) or "(none)"
    fact_block = "\n".join(f"- {f['fact']}" for f in world_facts[-30:]) or "(none)"
    sum_block = "\n".join(f"Ch.{s['number']}: {s['summary']}" for s in summaries[-4:]) or "(first chapter)"
    user = f"""Novel: "{project['title']}" ({project.get('genre','')})
Chapter {chapter_number} assigned beat: {beat}

MEMORY — characters:
{char_block}
MEMORY — world facts:
{fact_block}
PRIOR SUMMARIES:
{sum_block}

CHAPTER TO REVIEW:
{chapter_text[:16000]}"""
    raw = llm.complete(REVIEWER_SYSTEM, user,
                       temperature=config.REVIEW_TEMPERATURE, max_tokens=1200)
    data = LMStudioClient.extract_json(raw)
    if not isinstance(data, dict):
        return {"pass": False, "score": 0, "issues": ["Reviewer output unparsable"],
                "notes": raw[:300]}
    return {"pass": bool(data.get("pass", False)),
            "score": float(data.get("score", 0) or 0),
            "issues": [str(i) for i in (data.get("issues") or [])],
            "notes": str(data.get("notes", ""))}


REWRITER_SYSTEM = """You are a novel revision specialist. Rewrite the chapter so it fixes
every listed review issue while preserving everything that already works. Keep the same
scene coverage and length class. Output ONLY the revised chapter formatted as:
# <Title>
<prose paragraphs>"""


def rewrite_chapter(llm: LMStudioClient, chapter_title: str, chapter_text: str,
                    review: Dict[str, Any], beat: str) -> Dict[str, str]:
    issues = "\n".join(f"- {i}" for i in review.get("issues", [])) or "- general polish"
    user = f"""Beat that must be accomplished: {beat}
Review score: {review.get('score')} / 10
Reviewer notes: {review.get('notes','')}
Issues to fix:
{issues}

ORIGINAL CHAPTER:
# {chapter_title}

{chapter_text[:16000]}"""
    raw = llm.complete(REWRITER_SYSTEM, user,
                       temperature=max(0.5, config.GEN_TEMPERATURE - 0.2),
                       max_tokens=config.GEN_MAX_TOKENS)
    from generator import parse_chapter
    parsed = parse_chapter(raw, fallback_title=chapter_title)
    # keep original title unless model gave a better one
    parsed["title"] = parsed["title"] or chapter_title
    return parsed
