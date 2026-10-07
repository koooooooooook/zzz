"""Thin OpenAI-compatible client for the LM Studio local server."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

import config


class LMStudioClient:
    """Chat completion helper pointed at LM Studio's local server.

    Default endpoint: http://127.0.0.1:12340/v1  (LM Studio Developer tab).
    Default model id: "local-model" — change it in the UI if LM Studio
    reports a different loaded-model identifier.
    """

    def __init__(self, base_url: str = config.LMSTUDIO_BASE_URL,
                 api_key: str = config.LMSTUDIO_API_KEY,
                 model: str = config.DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    # -- core --------------------------------------------------------------
    def chat(self, messages: List[Dict[str, str]],
             temperature: float = config.GEN_TEMPERATURE,
             max_tokens: int = config.GEN_MAX_TOKENS,
             top_p: float = config.GEN_TOP_P) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return (resp.choices[0].message.content or "").strip()

    def complete(self, system: str, user: str, **kw: Any) -> str:
        return self.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], **kw)

    # -- utilities -----------------------------------------------------------
    def list_models(self) -> List[str]:
        try:
            return [m.id for m in self.client.models.list().data]
        except Exception:
            return []

    def ping(self) -> Dict[str, Any]:
        """Return a status dict used by the UI's connection test."""
        try:
            models = self.list_models()
            _ = self.complete("You are a health check.", "Reply with exactly: OK",
                              temperature=0.0, max_tokens=8)
            return {"ok": True, "models": models, "base_url": self.base_url}
        except Exception as exc:  # pragma: no cover - network dependent
            return {"ok": False, "error": str(exc), "base_url": self.base_url}

    @staticmethod
    def extract_json(text: str) -> Optional[Any]:
        """Best-effort extraction of a JSON object/array from model output."""
        # Try direct parse first.
        try:
            return json.loads(text)
        except Exception:
            pass
        # Fenced code block?
        fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
        if fence:
            try:
                return json.loads(fence.group(1))
            except Exception:
                pass
        # First balanced {...} or [...]
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except Exception:
                            break
        return None
