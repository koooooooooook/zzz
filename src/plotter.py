"""Automatic plotting: turn a premise into an act/beat outline."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from llm import LMStudioClient
import config


PLOTTER_SYSTEM = """You are a professional story architect for serialized fiction.
Given a novel premise, produce a beat-by-beat outline that can sustain a full-length
novel (about 100,000 characters). Each beat becomes one chapter.
Respond ONLY with JSON of this exact shape:
{
  "title": "...",
  "logline": "...",
  "beats": [
    {"chapter": 1, "beat": "one-sentence chapter goal / plot event", "arc": "setup|rising|climax|resolution"}
  ]
}
Rules: beats must form a coherent causal chain; escalate tension; end with a
satisfying resolution; 40-60 beats is typical."""


def generate_outline(llm: LMStudioClient, premise: str, genre: str = "",
                     target_chars: int = config.TARGET_CHARS,
                     chars_per_chapter: int = config.CHAPTER_TARGET_CHARS) -> Dict[str, Any]:
    n_beats = max(20, int(target_chars // max(500, chars_per_chapter)) + 8)
    user = (f"Premise:\n{premise}\n\n"
            f"Genre: {genre or 'unspecified'}\n"
            f"Produce exactly {n_beats} chapter beats "
            f"(each chapter ~{chars_per_chapter} characters, total ~{target_chars}).")
    raw = llm.complete(PLOTTER_SYSTEM, user, temperature=0.7, max_tokens=config.GEN_MAX_TOKENS)
    data = LMStudioClient.extract_json(raw)
    if not isinstance(data, dict) or not data.get("beats"):
        # Fallback deterministic skeleton so generation never dead-ends.
        data = {"title": "Untitled", "logline": premise[:200], "beats": []}
        arcs = ["setup"] * 5 + ["rising"] * (n_beats - 10) + ["climax"] * 3 + ["resolution"] * 2
        for i in range(1, n_beats + 1):
            arc = arcs[i - 1] if i - 1 < len(arcs) else "rising"
            data["beats"].append({"chapter": i,
                                  "beat": f"Chapter {i}: advance the {arc} phase of the story.",
                                  "arc": arc})
    # Normalize
    beats: List[Dict[str, Any]] = []
    for i, b in enumerate(data["beats"], start=1):
        beats.append({
            "chapter": int(b.get("chapter", i)),
            "beat": str(b.get("beat", "")).strip() or f"Chapter {i} beat",
            "arc": str(b.get("arc", "rising")).strip() or "rising",
        })
    data["beats"] = beats
    return data


def next_beat(outline: Dict[str, Any], chapter_number: int) -> str:
    beats = outline.get("beats", [])
    for b in beats:
        if b["chapter"] == chapter_number:
            return b["beat"]
    if chapter_number > len(beats):
        last = beats[-1]["beat"] if beats else "resolve remaining threads"
        return f"Continue past the planned outline — extend resolution from: {last}"
    return f"Chapter {chapter_number}: develop the story naturally."
