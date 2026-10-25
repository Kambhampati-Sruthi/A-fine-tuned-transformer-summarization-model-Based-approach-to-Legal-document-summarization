
import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text):
    """Simple transparent tokenization for lexical retrieval."""
    return re.findall(r"[a-zA-Z0-9]+(?:['’-][a-zA-Z0-9]+)*", str(text).lower())


def segment_document(text, max_words=400, overlap_words=60):
    """
    Create source-linked chunks.
    Offsets are character offsets in the normalized input text.
    """
    text = str(text)
    matches = list(re.finditer(r"\S+", text))

    if not matches:
        return []

    step = max_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be smaller than max_words")

    records = []
    chunk_index = 0

    for start_word in range(0, len(matches), step):
        end_word = min(start_word + max_words, len(matches))
        start_char = matches[start_word].start()
        end_char = matches[end_word - 1].end()

        records.append({
            "evidence_id": f"CHUNK-{chunk_index + 1:04d}",
            "section": f"Chunk {chunk_index + 1}",
            "title": "",
            "source_start_char": start_char,
            "source_end_char": end_char,
            "text": text[start_char:end_char],
        })

        chunk_index += 1

        if end_word == len(matches):
            break

    return records


def load_clause_csv(csv_path):
    """Load previously extracted clauses while retaining available identifiers."""
    df = pd.read_csv(csv_path)

    text_column = next(
        (c for c in ["clause_text", "text", "context", "source_text"]
         if c in df.columns),
        None
    )

    if text_column is None:
        raise ValueError(
            "Clause CSV must contain clause_text, text, context, or source_text."
        )

    records = []

    for i, row in df.iterrows():
        text = str(row[text_column])
        if not text.strip() or text.lower() == "nan":
            continue

        section = str(row.get("section", f"Row {i + 1}"))
        title = str(row.get("title", ""))

        records.append({
            "evidence_id": f"CLAUSE-{i + 1:04d}",
            "section": section,
            "title": title,
            "source_start_char": None,
            "source_end_char": None,
            "text": text,
        })

    return records


class EvidenceRetriever:
    """
    Hybrid lexical retriever combining BM25 and TF-IDF cosine similarity.

    Scores are relevance signals, not probabilities or proof of legal support.
    """

    def __init__(self, records, bm25_weight=0.5, tfidf_weight=0.5):
        self.records = [
            r for r in records if str(r.get("text", "")).strip()
        ]

        if not self.records:
            raise ValueError("No non-empty evidence records provided.")

        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight

        self.documents = [r["text"] for r in self.records]
        self.tokens = [tokenize(text) for text in self.documents]

        self.bm25 = BM25Okapi(self.tokens)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    @staticmethod
    def normalize_scores(scores):
        scores = scores.astype(float)
        low, high = scores.min(), scores.max()

        if high - low < 1e-12:
            return scores * 0.0

        return (scores - low) / (high - low)

    def search(self, query, top_k=5):
        query = str(query).strip()

        if not query:
            return []

        query_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        query_vector = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(
            query_vector, self.tfidf_matrix
        ).ravel()

        bm25_norm = self.normalize_scores(bm25_scores)
        tfidf_norm = self.normalize_scores(tfidf_scores)

        combined = (
            self.bm25_weight * bm25_norm
            + self.tfidf_weight * tfidf_norm
        )

        ranked = combined.argsort()[::-1][:max(1, top_k)]

        results = []

        for idx in ranked:
            record = self.records[int(idx)]
            results.append({
                **record,
                "bm25_score_normalized": float(bm25_norm[idx]),
                "tfidf_similarity": float(tfidf_scores[idx]),
                "retrieval_score": float(combined[idx]),
                "retrieval_method": "BM25 + TF-IDF",
                "requires_support_verification": True,
            })

        return results
