"""Heuristic key-fact extraction (parties, dates, durations, money, notice, law).
Every fact carries evidence chunk ids; values are copied verbatim from the source."""
from __future__ import annotations
import re
from ..schemas.models import Chunk, KeyFact
from ..ingestion.chunking import split_sentences

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE = (rf"(?:(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS}),?\s+\d{{4}}"
        r"|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})")
DUR = r"(?:\w+\s*\(\d+\)|\d+)\s*(?:calendar\s+|business\s+)?(?:days?|weeks?|months?|years?)"
MONEY = r"(?:[$€£₹]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand))?|(?:USD|EUR|GBP|INR)\s?\d[\d,]*(?:\.\d+)?)"

PARTY_RE = re.compile(
    r"(?:between|by and between)\s+(?P<a>[A-Z][\w&.,' -]{2,70}?)\s*(?:,[^()]{0,50})?\(.{0,40}?\)\s*,?\s*(?:and|&)\s+"
    r"(?P<b>[A-Z][\w&.,' -]{2,70}?)\s*(?:,[^()]{0,50})?(?=\s*(?:\(|,))")
EFFECTIVE_RE = re.compile(rf"(?:effective(?:\s+date)?|dated|commenc\w+|entered into)[^.]{{0,60}}?(?P<d>{DATE})", re.I)
LAW_RE = re.compile(r"governed by[^.]{0,60}?laws? of (?:the )?(?P<j>(?:State|Commonwealth|Republic) of [A-Z][\w ]+|[A-Z][\w]+(?: [A-Z][\w]+)*)", re.I)
TERM_RE = re.compile(rf"(?:initial\s+)?term of\s+(?P<t>{DUR})", re.I)
NOTICE_RE = re.compile(rf"(?P<n>{DUR})['’]?\s*(?:prior\s+|advance\s+)?(?:written\s+)?notice", re.I)
PAY_RE = re.compile(rf"(?:within|no later than)\s+(?P<p>{DUR})\s+(?:of|after|from)\s+(?:the\s+)?(?:receipt|date|invoice)", re.I)
LATE_RE = re.compile(r"(?P<r>\d+(?:\.\d+)?\s?%)\s*(?:per\s+(?:month|annum|year)|monthly|annually)", re.I)
RENEW_RE = re.compile(r"(?:automatically\s+renew\w*|auto-?renew\w*|renew\w*\s+automatically)", re.I)
FEE_RE = re.compile(rf"(?P<m>{MONEY})", re.I)


def _find(chunks, rx, group, field, label=None, first_only=True, sentence_filter=None):
    facts = []
    for c in chunks:
        for s in split_sentences(c.text):
            if sentence_filter and not sentence_filter.search(s):
                continue
            m = rx.search(s)
            if m:
                facts.append(KeyFact(field=label or field, value=m.group(group).strip(" ,"), evidence_ids=[c.chunk_id]))
                if first_only:
                    return facts
    return facts


def extract_facts(chunks: list[Chunk]) -> list[KeyFact]:
    facts: list[KeyFact] = []
    head = chunks[:3]
    for c in head:
        m = PARTY_RE.search(c.text.replace("\n", " "))
        if m:
            facts.append(KeyFact(field="Parties", value=f"{m.group('a').strip(' ,')} and {m.group('b').strip(' ,')}",
                                 evidence_ids=[c.chunk_id]))
            break
    facts += _find(head, EFFECTIVE_RE, "d", "Effective date")
    facts += _find(chunks, TERM_RE, "t", "Term / duration")
    facts += _find(chunks, NOTICE_RE, "n", "Notice period",
                   sentence_filter=re.compile(r"terminat|renew|cancel|breach", re.I))
    facts += _find(chunks, PAY_RE, "p", "Payment deadline")
    facts += _find(chunks, LATE_RE, "r", "Late fee / interest")
    facts += _find(chunks, LAW_RE, "j", "Governing law")
    for c in chunks:
        if RENEW_RE.search(c.text):
            facts.append(KeyFact(field="Auto-renewal", value="Appears to renew automatically (see clause)", evidence_ids=[c.chunk_id]))
            break
    amounts = []
    for c in chunks:
        for m in FEE_RE.finditer(c.text):
            amounts.append((m.group("m"), c.chunk_id))
    for val, cid in amounts[:4]:
        facts.append(KeyFact(field="Amount mentioned", value=val, evidence_ids=[cid]))
    return facts
