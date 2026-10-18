"""Source-linked extractive summarizer: TF-IDF centrality + legal cue boosts + MMR de-duplication.

Used as (a) the E0 baseline, (b) the compression stage of the hybrid pipeline, and
(c) the always-available fallback when no neural model is installed.
"""
from __future__ import annotations
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from ..ingestion.chunking import split_sentences
from ..schemas.models import Chunk

CUES = re.compile(
    r"\b(shall|must|may not|will not|agrees? to|terminat\w+|renew\w*|notice|within \d+|"
    r"\d+\s*(?:days?|months?|years?)|indemnif\w+|liabilit\w+|confidential\w*|governed by|"
    r"payment|fees?|invoice|late|breach|warrant\w+|assign\w*|exclusiv\w+|non-?compet\w+)\b", re.I)
NUMERIC = re.compile(r"(\$|€|£|₹)\s?\d|\b\d+(?:\.\d+)?\s?%|\b\d{4}\b|\b\d+\s*(?:days?|months?|years?)\b", re.I)


def candidate_sentences(chunks: list[Chunk]) -> list[tuple[str, str]]:
    out = []
    for c in chunks:
        for s in split_sentences(c.text):
            n = len(s.split())
            if 7 <= n <= 70 and not s.isupper():
                out.append((s, c.chunk_id))
    return out


def rank_sentences(sents: list[tuple[str, str]], first_chunk_id: str | None = None) -> np.ndarray:
    if len(sents) < 2:
        return np.ones(len(sents))
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([s for s, _ in sents])
    centroid = np.asarray(X.mean(axis=0))
    cent = np.asarray(X @ centroid.T).ravel()
    cent = cent / (cent.max() or 1)
    cue = np.array([min(len(CUES.findall(s)), 4) / 4 + (0.25 if NUMERIC.search(s) else 0) for s, _ in sents])
    pos = np.array([0.3 if cid == first_chunk_id else 0.0 for _, cid in sents])
    return 0.5 * cent + 0.4 * cue + pos


def select_mmr(sents, scores, n: int, lam: float = 0.7) -> list[int]:
    if not sents:
        return []
    vec = TfidfVectorizer(stop_words="english").fit([s for s, _ in sents])
    X = vec.transform([s for s, _ in sents])
    sim = (X @ X.T).toarray()
    chosen: list[int] = []
    cand = list(range(len(sents)))
    while cand and len(chosen) < n:
        best = max(cand, key=lambda i: lam * scores[i] - (1 - lam) * (max(sim[i][j] for j in chosen) if chosen else 0))
        chosen.append(best)
        cand.remove(best)
    return sorted(chosen)  # restore document order


def summarize_extractive(chunks: list[Chunk], n_exec: int = 8, per_section: int = 2):
    """Returns (executive: list[(sentence, chunk_id)], sections: list[dict])."""
    sents = candidate_sentences(chunks)
    scores = rank_sentences(sents, chunks[0].chunk_id if chunks else None)
    idx = select_mmr(sents, scores, n_exec)
    executive = [sents[i] for i in idx]

    groups: dict[str, list[Chunk]] = {}
    for c in chunks:
        groups.setdefault(c.clause_id or c.heading or c.chunk_id, []).append(c)
    sections = []
    for key, cs in groups.items():
        ss = candidate_sentences(cs)
        if not ss:
            continue
        sc = rank_sentences(ss)
        pick = select_mmr(ss, sc, per_section)
        sections.append({
            "heading": (f"§{cs[0].clause_id} " if cs[0].clause_id else "") + (cs[0].heading or ""),
            "summary": " ".join(ss[i][0] for i in pick),
            "evidence_ids": sorted({ss[i][1] for i in pick}),
        })
    return executive, sections
