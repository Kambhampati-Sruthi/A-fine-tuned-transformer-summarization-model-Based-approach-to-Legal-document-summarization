"""Evaluation metrics. No single score establishes legal factual correctness - report several."""
from __future__ import annotations
import re
from collections import Counter

_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())


def _ngrams(t, n):
    return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))


def rouge_n(pred: str, ref: str, n: int = 1) -> dict:
    p, r = _ngrams(_tok(pred), n), _ngrams(_tok(ref), n)
    ov = sum((p & r).values())
    prec, rec = ov / max(1, sum(p.values())), ov / max(1, sum(r.values()))
    return {"p": prec, "r": rec, "f": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def rouge_l(pred: str, ref: str) -> dict:
    a, b = _tok(pred), _tok(ref)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[-1][-1]
    prec, rec = l / max(1, len(a)), l / max(1, len(b))
    return {"p": prec, "r": rec, "f": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def rouge_all(pred, ref):
    """Self-contained ROUGE-1/2/L F1 (use `rouge-score` for publication numbers; this avoids a hard dependency)."""
    return {"rouge1": rouge_n(pred, ref, 1)["f"], "rouge2": rouge_n(pred, ref, 2)["f"], "rougeL": rouge_l(pred, ref)["f"]}


def bertscore(preds, refs, lang="en"):
    from bert_score import score  # optional dependency
    P, R, F = score(preds, refs, lang=lang, verbose=False)
    return float(F.mean())


def key_term_coverage(summary: str, terms: list[str]) -> float:
    s = summary.lower()
    return sum(t.lower() in s for t in terms) / max(1, len(terms))


def recall_at_k(ranked_clause_ids: list[list[str]], gold: list[str], k: int) -> float:
    return sum(g in r[:k] for r, g in zip(ranked_clause_ids, gold)) / max(1, len(gold))


def mrr(ranked_clause_ids: list[list[str]], gold: list[str]) -> float:
    s = 0.0
    for r, g in zip(ranked_clause_ids, gold):
        if g in r:
            s += 1 / (r.index(g) + 1)
    return s / max(1, len(gold))


RUBRIC = {"incorrect": 0, "unsupported": 1, "partial": 2, "supported": 3}
VERIFIER_TO_RUBRIC = {"contradicted": 0, "not_verifiable": 1, "partially_supported": 2, "supported": 3}


def factuality_from_claims(claims) -> dict:
    """Automatic proxy for the 0-3 manual rubric (use real human scores for headline results)."""
    n = max(1, len(claims))
    scores = [VERIFIER_TO_RUBRIC[c.support] for c in claims]
    return {"mean_rubric": sum(scores) / n, "pct_supported": sum(s == 3 for s in scores) / n,
            "pct_contradicted": sum(s == 0 for s in scores) / n}
