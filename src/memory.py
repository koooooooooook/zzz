"""SQLite structured memory: characters, world facts, events, chapters."""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    role TEXT DEFAULT '',
    description TEXT DEFAULT '',
    traits TEXT DEFAULT '[]',
    status TEXT DEFAULT 'alive',
    location TEXT DEFAULT '',
    goals TEXT DEFAULT '',
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS world_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    fact TEXT NOT NULL,
    source_chapter INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    chapter_number INTEGER NOT NULL,
    event TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    number INTEGER NOT NULL,
    title TEXT DEFAULT '',
    content TEXT NOT NULL,
    summary TEXT DEFAULT '',
    char_count INTEGER DEFAULT 0,
    review_pass INTEGER DEFAULT 1,
    UNIQUE(project_id, number)
);
"""


class MemoryDB:
    """Thread-safe SQLite wrapper.

    Gradio handlers and the background generation thread touch the same
    instance, so connections are kept per-thread (threading.local) and all
    writes are serialized with a lock.
    """

    def __init__(self, path: str = config.SQLITE_PATH):
        self.path = path
        self._local = threading.local()
        self._lock = threading.RLock()
        with self._conn_ctx() as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    # -- connection handling ---------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        """Connection bound to the *current* thread (created on first use)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _conn_ctx(self):
        return self.conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


    def upsert_character(self, *a, **kw):
        with self._lock:
            return getattr(self, '_upsert_character_impl')(*a, **kw)

    def add_world_fact(self, *a, **kw):
        with self._lock:
            return getattr(self, '_add_world_fact_impl')(*a, **kw)

    def add_event(self, *a, **kw):
        with self._lock:
            return getattr(self, '_add_event_impl')(*a, **kw)

    def save_chapter(self, *a, **kw):
        with self._lock:
            return getattr(self, '_save_chapter_impl')(*a, **kw)

    # -- characters ----------------------------------------------------------
    def _upsert_character_impl(self, project_id: str, name: str, **fields: Any) -> None:
        name = (name or "").strip()
        if not name:
            return
        row = self.conn.execute(
            "SELECT * FROM characters WHERE project_id=? AND name=?",
            (project_id, name)).fetchone()
        traits = fields.get("traits")
        if isinstance(traits, list):
            traits = json.dumps(traits, ensure_ascii=False)
        if row is None:
            self.conn.execute(
                """INSERT INTO characters
                   (project_id,name,role,description,traits,status,location,goals)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (project_id, name,
                 fields.get("role", ""), fields.get("description", ""),
                 traits or "[]", fields.get("status", "alive"),
                 fields.get("location", ""), fields.get("goals", "")))
        else:
            sets, vals = [], []
            for col in ("role", "description", "traits", "status", "location", "goals"):
                if col in fields and fields[col] not in (None, ""):
                    v = fields[col]
                    if isinstance(v, list):
                        v = json.dumps(v, ensure_ascii=False)
                    sets.append(f"{col}=?")
                    vals.append(v)
            if sets:
                vals.extend([project_id, name])
                self.conn.execute(
                    f"UPDATE characters SET {','.join(sets)} WHERE project_id=? AND name=?",
                    vals)
        self.conn.commit()

    def get_characters(self, project_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM characters WHERE project_id=? ORDER BY name",
            (project_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["traits"] = json.loads(d.get("traits") or "[]")
            except Exception:
                d["traits"] = []
            out.append(d)
        return out

    # -- world facts -----------------------------------------------------------
    def _add_world_fact_impl(self, project_id: str, fact: str,
                       category: str = "general", chapter: int = 0) -> None:
        fact = (fact or "").strip()
        if not fact:
            return
        exists = self.conn.execute(
            "SELECT 1 FROM world_facts WHERE project_id=? AND fact=?",
            (project_id, fact)).fetchone()
        if not exists:
            self.conn.execute(
                "INSERT INTO world_facts(project_id,category,fact,source_chapter) VALUES(?,?,?,?)",
                (project_id, category, fact, chapter))
            self.conn.commit()

    def get_world_facts(self, project_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM world_facts WHERE project_id=? ORDER BY id",
            (project_id,)).fetchall()
        return [dict(r) for r in rows]

    # -- events ------------------------------------------------------------------
    def _add_event_impl(self, project_id: str, chapter_number: int, event: str) -> None:
        event = (event or "").strip()
        if event:
            self.conn.execute(
                "INSERT INTO events(project_id,chapter_number,event) VALUES(?,?,?)",
                (project_id, chapter_number, event))
            self.conn.commit()

    def recent_events(self, project_id: str, limit: int = 15) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE project_id=? ORDER BY chapter_number DESC, id DESC LIMIT ?",
            (project_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # -- chapters -----------------------------------------------------------------
    def _save_chapter_impl(self, project_id: str, number: int, title: str,
                     content: str, summary: str, review_pass: bool) -> None:
        self.conn.execute(
            """INSERT INTO chapters(project_id,number,title,content,summary,char_count,review_pass)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(project_id,number) DO UPDATE SET
                 title=excluded.title, content=excluded.content,
                 summary=excluded.summary, char_count=excluded.char_count,
                 review_pass=excluded.review_pass""",
            (project_id, number, title, content, summary, len(content),
             1 if review_pass else 0))
        self.conn.commit()

    def get_chapter(self, project_id: str, number: int) -> Optional[Dict[str, Any]]:
        r = self.conn.execute(
            "SELECT * FROM chapters WHERE project_id=? AND number=?",
            (project_id, number)).fetchone()
        return dict(r) if r else None

    def chapter_summaries(self, project_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT number,title,summary,char_count FROM chapters WHERE project_id=? ORDER BY number",
            (project_id,)).fetchall()
        return [dict(r) for r in rows]

    def total_chars(self, project_id: str) -> int:
        r = self.conn.execute(
            "SELECT COALESCE(SUM(char_count),0) AS t FROM chapters WHERE project_id=?",
            (project_id,)).fetchone()
        return int(r["t"])

    def last_chapter_number(self, project_id: str) -> int:
        r = self.conn.execute(
            "SELECT COALESCE(MAX(number),0) AS m FROM chapters WHERE project_id=?",
            (project_id,)).fetchone()
        return int(r["m"])

    def close(self) -> None:
        self.conn.close()
