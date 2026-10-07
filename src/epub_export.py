"""EPUB export of all completed chapters via ebooklib."""
from __future__ import annotations

import os
import re
from typing import Any, Dict

from ebooklib import epub

import config


def _html_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def paragraphs_to_html(text: str) -> str:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out = []
    for b in blocks:
        # skip separator / summary artifacts from stored md files
        if b in ("---",) or b.startswith("Summary:"):
            continue
        inline = _html_escape(b).replace("\n", "<br/>\n")
        out.append(f"<p>{inline}</p>")
    return "\n".join(out)


def export_epub(project: Dict[str, Any], chapters: Dict[int, Dict[str, str]],
                out_path: str | None = None) -> str:
    """chapters: {number: {"title":..., "content":...}} — must be sorted internally."""
    pid = project["project_id"]
    out_path = out_path or os.path.join(config.OUTPUT_DIR, f"{pid}.epub")

    book = epub.EpubBook()
    book.set_identifier(f"local-novel-ai-{pid}")
    book.set_title(project.get("title") or pid)
    book.set_language("en")
    book.add_author("Local Novel AI")

    spine = ["nav"]
    toc = []
    for num in sorted(chapters):
        ch = chapters[num]
        fname = f"chapter_{num:03d}.xhtml"
        item = epub.EpubItem(uid=f"ch{num}", file_name=fname,
                             media_type="application/xhtml+xml",
                             content=(
                                 '<?xml version="1.0" encoding="utf-8"?>'
                                 '<html xmlns="http://www.w3.org/1999/xhtml">'
                                 f'<head><title>Chapter {num}</title></head><body>'
                                 f'<h1>Chapter {num}: {_html_escape(ch["title"])}</h1>'
                                 + paragraphs_to_html(ch["content"])
                                 + '</body></html>'))
        book.add_item(item)
        spine.append(item)
        toc.append(epub.Link(fname, f"Chapter {num}: {ch['title']}", f"ch{num}"))

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    book.toc = toc
    epub.write_epub(out_path, book, {})
    return out_path


def load_chapters_from_db(memory, project_id: str) -> Dict[int, Dict[str, str]]:
    rows = memory.chapter_summaries(project_id)
    out: Dict[int, Dict[str, str]] = {}
    for r in rows:
        full = memory.get_chapter(project_id, r["number"])
        if full:
            out[r["number"]] = {"title": full["title"], "content": full["content"]}
    return out
