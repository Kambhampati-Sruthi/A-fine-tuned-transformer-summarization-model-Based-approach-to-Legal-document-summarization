"""CUAD-inspired clause taxonomy (subset). Rule-based classifier; swap in a fine-tuned
Legal-BERT / LEDGAR classifier later behind the same function signature."""
from __future__ import annotations
import re
from ..schemas.models import Chunk, ClauseHit

TAXONOMY: dict[str, list[str]] = {
    "Governing Law": [r"governed by", r"laws? of the", r"construed in accordance"],
    "Termination": [r"terminat\w+", r"expir\w+ of this agreement"],
    "Termination for Convenience": [r"terminate[^.]{0,80}(for any reason|without cause|for convenience|at any time)"],
    "Notice Period": [r"\b\w+\s*(?:\(\d+\))?\s*(?:calendar |business )?days'? (?:prior |advance )?(?:written )?notice", r"written notice"],
    "Renewal / Term": [r"auto\w*[- ]?renew", r"renew\w*", r"initial term", r"term of this agreement"],
    "Payment": [r"\bpay(?:ment|able|s)?\b", r"invoice", r"\bfees?\b", r"compensation"],
    "Late Fees / Interest": [r"late (?:fee|charge|payment)", r"interest (?:at|of)[^.]{0,40}%", r"per (?:month|annum)"],
    "Confidentiality": [r"confidential\w*", r"non-?disclosure", r"proprietary information"],
    "IP Ownership / Assignment": [r"intellectual property", r"work[- ]product", r"work made for hire", r"assigns? (?:all )?(?:right|title)"],
    "Indemnification": [r"indemnif\w+", r"hold harmless"],
    "Limitation of Liability": [r"limitation of liability", r"in no event shall[^.]{0,80}liab", r"aggregate liability", r"consequential damages"],
    "Non-Compete / Exclusivity": [r"non-?compet\w+", r"exclusiv\w+", r"shall not (?:directly or indirectly )?(?:engage|compete)"],
    "Assignment": [r"may not assign", r"assign(?:ment)? of this agreement", r"without (?:the )?(?:prior )?(?:written )?consent"],
    "Dispute Resolution": [r"arbitrat\w+", r"jurisdiction", r"venue", r"dispute"],
    "Force Majeure": [r"force majeure", r"beyond (?:its|their) reasonable control"],
    "Warranty": [r"warrant\w+", r"as is"],
    "Amendment / Modification": [r"amend\w+", r"modif\w+ (?:this|the) agreement", r"sole discretion[^.]{0,40}(?:change|modify)"],
    "Insurance": [r"insurance"],
    "Audit Rights": [r"audit"],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in TAXONOMY.items()}


def classify_chunk(c: Chunk) -> list[tuple[str, float]]:
    out = []
    head = (c.heading or "").lower()
    for label, pats in _COMPILED.items():
        hits = sum(1 for p in pats if p.search(c.text))
        title_hit = any(p.search(head) for p in pats)
        if hits or title_hit:
            conf = min(1.0, 0.35 * hits + (0.5 if title_hit else 0.0))
            if conf >= 0.35:
                out.append((label, round(conf, 2)))
    return sorted(out, key=lambda x: -x[1])


def detect_clauses(chunks: list[Chunk]) -> list[ClauseHit]:
    hits = []
    for c in chunks:
        labels = classify_chunk(c)
        c.clause_types = [l for l, _ in labels]
        for label, conf in labels:
            hits.append(ClauseHit(clause_type=label, chunk_id=c.chunk_id, ref=c.ref(),
                                  snippet=c.text[:280], confidence=conf))
    return hits
