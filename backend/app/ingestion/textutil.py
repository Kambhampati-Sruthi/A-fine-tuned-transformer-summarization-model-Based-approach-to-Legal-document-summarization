"""Shared text heuristics (heading detection) used by extraction and chunking."""
import re

HEADING_RE = re.compile(
    r"^\s*(?:(?P<art>(?:ARTICLE|Article|SECTION|Section|CLAUSE|Clause)\s+[IVXLC\d]+(?:\.\d+)*)[.:\-\s]*"
    r"|(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+)"
    r"(?P<title>[A-Z][^\n]{0,90})?$"
)


def _is_heading(line: str):
    s = line.strip()
    if not s or len(s) > 110:
        return None
    m = HEADING_RE.match(s)
    if not m:
        return None
    title = (m.group("title") or "").strip()
    ident = (m.group("art") or m.group("num") or "").strip()
    # a numbered line that reads like a full sentence is a numbered paragraph, not a heading
    if m.group("num") and title and title.endswith(".") and len(title.split()) > 8:
        return None
    clause_id = re.sub(r"^(?i:article|section|clause)\s+", "", ident)
    return clause_id, title or s
