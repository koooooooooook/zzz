"""Mock LM Studio server (OpenAI-compatible) for offline end-to-end testing.

Serves POST /v1/chat/completions and GET /v1/models on 127.0.0.1:12340.
Behaviour is prompt-aware so the full pipeline can be exercised without a GPU:
  - plotter   -> JSON outline with N beats
  - reviewer  -> first review FAILS (forces the auto-rewrite path), then passes
  - extractor -> JSON character/world/event memory
  - summarizer-> one-paragraph summary
  - writer    -> markdown chapter long enough to pass min-length checks
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_lock = threading.Lock()
_review_counts: dict[str, int] = {}


def _user_text(body: dict) -> str:
    parts = []
    for m in body.get("messages", []):
        if m.get("role") == "user":
            parts.append(m.get("content", ""))
    return "\n".join(parts)


def _chapter_md(num: int, tag: str = "") -> str:
    title = f"The Salt Hour {tag}".strip()
    para = (
        "The lamp guttered twice before Kael understood that the sea was answering it, "
        "each flash of the great light returning as a pale pulse far out on the water. "
        "He pressed his forehead to the cold glass and counted the intervals, because "
        "counting was the only ritual that had ever kept the panic at bay. Somewhere "
        "below, in the gallery where the foghorn should have been, something began to "
        "breathe in time with the beacon."
    )
    paras = "\n\n".join([para] * 14)
    return f"# {title}\n\n{paras}\n"


def _respond(body: dict) -> str:
    user = _user_text(body)
    msgs = body.get("messages", [])
    system = msgs[0]["content"] if msgs else ""

    # Plotter
    if "story architect" in system:
        m = re.search(r"exactly (\d+) chapter beats", user)
        n = int(m.group(1)) if m else 48
        beats = []
        arcs = ["setup"] * 5 + ["rising"] * (n - 10) + ["climax"] * 3 + ["resolution"] * 2
        for i in range(1, n + 1):
            arc = arcs[i - 1] if i - 1 < len(arcs) else "rising"
            beats.append({"chapter": i,
                          "beat": f"Chapter {i}: advance the {arc} phase of the story.",
                          "arc": arc})
        return json.dumps({"title": "The Salt Hour",
                           "logline": "A lighthouse keeper discovers the light answers back.",
                           "beats": beats})

    # Reviewer
    if "continuity and craft editor" in system:
        ch = re.search(r"Chapter (\d+) assigned beat", user)
        key = ch.group(1) if ch else "?"
        with _lock:
            seen = _review_counts.get(key, 0)
            _review_counts[key] = seen + 1
        if seen == 0:  # force one rewrite round the first time each chapter is reviewed
            return json.dumps({"pass": False, "score": 5.5,
                               "issues": ["Beat not yet accomplished; add the sea's reply."],
                               "notes": "Promising but incomplete."})
        return json.dumps({"pass": True, "score": 8.5, "issues": [],
                           "notes": "Continuity holds; hook lands."})

    # Memory extractor
    if "extract story state" in system:
        return json.dumps({
            "characters": [
                {"name": "Kael", "role": "lighthouse keeper", "status": "alive",
                 "description": "Ritualistic, insomniac keeper of the Gullspire light.",
                 "traits": ["observant", "stubborn"], "location": "Gullspire Lighthouse",
                 "goals": "Understand the responding light"},
                {"name": "Maren", "role": "harbourmaster", "status": "alive",
                 "description": "Keeps the tide ledger and everyone's secrets.",
                 "traits": ["guarded"], "location": "Port Sallow", "goals": "Keep the truth ashore"},
            ],
            "world_facts": [
                {"category": "rule", "fact": "The Gullspire beacon pulses back when the sea signals."},
                {"category": "location", "fact": "Port Sallow sits below Gullspire on a black-sand coast."},
            ],
            "events": ["The sea answered the beacon for the first time in living memory."],
        })

    # Summarizer
    if "summarize novel chapters" in system:
        return ("Kael confirmed the beacon was being answered from offshore, began "
                "logging the intervals, and heard breathing behind the foghorn door.")

    # Rewriter (revision specialist) — must produce a fresh, long-enough chapter
    if "revision specialist" in system:
        m = re.search(r"Chapter (\d+)", user)
        num = int(m.group(1)) if m else 1
        return _chapter_md(num, tag="(revised)")

    # Health check
    if "health check" in system.lower():
        return "OK"

    # Writer default
    m = re.search(r"Write Chapter (\d+)", user)
    num = int(m.group(1)) if m else 1
    return _chapter_md(num)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list",
                             "data": [{"id": "local-model", "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        content = _respond(body)
        self._send(200, {
            "id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
            "model": body.get("model", "local-model"),
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


def serve(port: int = 12340) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    srv = serve()
    print("Mock LM Studio listening on http://127.0.0.1:12340/v1 (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
