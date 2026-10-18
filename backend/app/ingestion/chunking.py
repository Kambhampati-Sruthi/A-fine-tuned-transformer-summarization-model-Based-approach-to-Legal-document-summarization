"""Clause/section-aware chunking that preserves page, clause id and char offsets."""
from __future__ import annotations
import re
from ..schemas.models import Chunk
from .extract import ExtractedDoc
from .textutil import _is_heading

SENT_RE = re.compile(r'(?<=[.;:!?])\s+(?=[A-Z("“\d])')


ABBREV = re.compile(r"\b(Inc|Ltd|Co|Corp|LLC|LLP|Pvt|No|Nos|Sec|Art|Dr|Mr|Mrs|Ms|St|vs|v|etc|e\.g|i\.e|Cl)\.", re.I)


def split_sentences(text: str) -> list[str]:
    """Sentence split that keeps legal abbreviations (Inc., Co., No.) intact and treats line breaks
    (headings, numbered items) as boundaries so headings never fuse with the next sentence."""
    out = []
    for line in text.split("\n"):
        line = re.sub(r"^(\s*\d{1,2}(?:\.\d{1,2})*)\.(?=\s)", lambda m: m.group(1) + "\u2024", line)  # "3. Fees"
        line = ABBREV.sub(lambda m: m.group(0)[:-1] + "\u2024", line)      # protect the dot
        for s in SENT_RE.split(line):
            s = s.replace("\u2024", ".").strip()
            if s:
                out.append(s)
    return out


def _window(text: str, max_words: int, overlap: int) -> list[tuple[str, int]]:
    """Split long clause text on sentence boundaries; returns (piece, char_offset_in_text)."""
    if len(text.split()) <= max_words:
        return [(text, 0)]
    pieces, cur, cur_w, off, cursor = [], [], 0, 0, 0
    for s in split_sentences(text):
        w = len(s.split())
        idx = text.find(s[:40], cursor)
        if cur and cur_w + w > max_words:
            pieces.append((" ".join(cur), off))
            tail = " ".join(" ".join(cur).split()[-overlap:]) if overlap else ""
            cur, cur_w = ([tail] if tail else []), len(tail.split())
            off = max(idx, 0)
        cur.append(s)
        cur_w += w
        if idx >= 0:
            cursor = idx + len(s)
    if cur:
        pieces.append((" ".join(cur), off))
    return pieces


def chunk_document(doc: ExtractedDoc, doc_id: str, max_words=220, overlap=30) -> list[Chunk]:
    # 1) cut the text into clause units on heading lines, tracking offsets
    units, cur = [], {"clause_id": None, "heading": "Preamble", "start": 0, "lines": []}
    pos = 0
    for line in doc.text.split("\n"):
        h = _is_heading(line)
        if h and cur["lines"]:
            cur["end"] = pos
            units.append(cur)
            cur = {"clause_id": h[0], "heading": h[1], "start": pos, "lines": []}
        elif h:
            cur.update(clause_id=h[0], heading=h[1], start=pos)
        cur["lines"].append(line)
        pos += len(line) + 1
    cur["end"] = len(doc.text)
    units.append(cur)

    # 2) enforce size limit inside each clause unit
    chunks: list[Chunk] = []
    for u in units:
        body = "\n".join(u["lines"]).strip()
        if len(body.split()) < 3:
            continue
        base = doc.text.find(body[:60], u["start"])
        base = u["start"] if base < 0 else base
        for piece, off in _window(body, max_words, overlap):
            s = base + off
            e = s + len(piece)
            chunks.append(Chunk(
                chunk_id=f"c{len(chunks):04d}", doc_id=doc_id, clause_id=u["clause_id"],
                heading=u["heading"], text=piece, page=doc.page_at(s),
                page_end=doc.page_at(max(e - 1, s)), start=s, end=e,
            ))
    return chunks
