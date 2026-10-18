"""Hybrid retrieval: BM25 (exact legal terms, section numbers) + dense vectors (semantic),
fused with Reciprocal Rank Fusion, optional cross-encoder rerank.

Legal retrieval needs *precise* snippets (LegalBench-RAG, Pipitone & Chaudhari 2024), so we
retrieve small clause-bounded chunks and keep clause id / page / offsets on every hit.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from ..schemas.models import Chunk

STOP = set("a an and are as at be by for from has have in is it its of on or that the their this to was were will with shall any such".split())
SUFFIXES = ("ations", "ation", "ating", "ments", "ment", "ions", "ion", "ings", "ing", "ates", "ated", "ate", "ed", "es", "ly", "s")
TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
SECTION_Q = re.compile(r"(?:section|clause|article|§)\s*([0-9]+(?:\.[0-9]+)*)", re.I)


def stem(w: str) -> str:
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOP]


class TfidfEmbedder:
    """Dependency-light stand-in for a neural embedder (word+bigram TF-IDF, L2-normalised)."""
    name = "tfidf-fallback"

    def fit(self, texts):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), sublinear_tf=True,
                                   max_features=20000, token_pattern=None)
        self.vec.fit(texts)
        return self

    def encode(self, texts, is_query=False):
        return self.vec.transform(texts).toarray().astype("float32")


class SBertEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.m = SentenceTransformer(model_name)

    def fit(self, texts):
        return self

    def encode(self, texts, is_query=False):
        return self.m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False).astype("float32")


def make_embedder(model_name: str):
    if model_name:
        try:
            return SBertEmbedder(model_name)
        except Exception:
            pass
    return TfidfEmbedder()


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder=None, reranker=None, rrf_k: int = 60):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.rrf_k = rrf_k
        self.reranker = reranker
        texts = [f"{c.heading or ''}. {c.text}" for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) or ["∅"] for t in texts])
        self.embedder = (embedder or TfidfEmbedder()).fit(texts)
        self.emb = self.embedder.encode(texts)
        self._faiss = None
        try:  # FAISS if installed; exact numpy search otherwise (identical results at this scale)
            import faiss
            self._faiss = faiss.IndexFlatIP(self.emb.shape[1])
            self._faiss.add(np.ascontiguousarray(self.emb))
        except Exception:
            pass

    # ---------------------------------------------------------------- search
    def _dense_rank(self, query: str, n: int) -> list[int]:
        q = self.embedder.encode([query], is_query=True)
        if self._faiss is not None:
            _, I = self._faiss.search(np.ascontiguousarray(q), min(n, len(self.chunks)))
            return [int(i) for i in I[0] if i >= 0]
        return list(np.argsort(-(self.emb @ q[0]))[:n])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        n = min(max(k * 4, 20), len(self.chunks))
        bm = self.bm25.get_scores(tokenize(query) or ["∅"])
        bm_rank = list(np.argsort(-bm)[:n])
        dn_rank = self._dense_rank(query, n)
        fused: dict[int, float] = {}
        for rank_list in (bm_rank, dn_rank):
            for r, i in enumerate(rank_list):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (self.rrf_k + r + 1)
        for m in SECTION_Q.finditer(query):  # exact clause-number lookups ("section 12.2")
            for i, c in enumerate(self.chunks):
                if c.clause_id == m.group(1):
                    fused[i] = fused.get(i, 0.0) + 0.05
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[: max(k * 2, k)]
        hits = [(self.chunks[i], s) for i, s in ranked]
        if self.reranker is not None and hits:
            sc = self.reranker.predict([(query, c.text) for c, _ in hits])
            hits = [h for _, h in sorted(zip(-np.asarray(sc), hits), key=lambda z: z[0])]
        return hits[:k]

    @staticmethod
    def lexical_confidence(query: str, text: str) -> float:
        q = set(tokenize(query))
        return len(q & set(tokenize(text))) / len(q) if q else 0.0

    # ----------------------------------------------------------- persistence
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "chunks.json").write_text(json.dumps([c.model_dump() for c in self.chunks]))

    @staticmethod
    def load_chunks(path: Path) -> list[Chunk]:
        return [Chunk(**d) for d in json.loads((path / "chunks.json").read_text())]


def make_reranker(model_name: str):
    if not model_name:
        return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception:
        return None
