"""Claim verification (SummaC-style: sentence-level support against source passages).

Default is a lexical + numeric/date-consistency verifier that needs no model. If NLI_MODEL is set
(an MNLI-style cross-encoder), entailment/contradiction probabilities are used and combined with the
numeric check, because ROUGE-like similarity cannot detect a changed number or deadline."""
from __future__ import annotations
import re
from ..ingestion.chunking import split_sentences
from ..rag.retriever import tokenize

NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split())}
NUM_WORDS.update({"thirty": "30", "forty": "40", "sixty": "60", "ninety": "90", "hundred": "100"})
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
NEG_RE = re.compile(r"\b(?:not|no|never|neither|nor|without|cannot|n't)\b", re.I)


def numbers(text: str) -> set[str]:
    t = text.lower()
    out = {n.replace(",", "") for n in NUM_RE.findall(t)}
    out |= {NUM_WORDS[w] for w in re.findall(r"[a-z]+", t) if w in NUM_WORDS}
    return out


class Verifier:
    def __init__(self, nli_model: str = ""):
        self._nli = None
        if nli_model:
            try:
                from transformers import pipeline
                self._nli = pipeline("text-classification", model=nli_model, top_k=None)
            except Exception:
                self._nli = None

    def _lexical(self, claim: str, evidence: list[str]):
        ctoks = set(tokenize(claim))
        if not ctoks or not evidence:
            return "not_verifiable", 0.0, ""
        best_s, best_cov = "", 0.0
        for ev in evidence:
            for s in split_sentences(ev) or [ev]:
                cov = len(ctoks & set(tokenize(s))) / len(ctoks)
                if cov > best_cov:
                    best_s, best_cov = s, cov
        ev_nums = numbers(" ".join(evidence))
        missing = numbers(claim) - ev_nums
        neg_mismatch = bool(NEG_RE.search(claim)) != bool(NEG_RE.search(best_s)) and best_cov >= 0.5
        if missing and best_cov >= 0.45:
            return "contradicted", best_cov, best_s       # same topic but a figure/date differs
        if neg_mismatch:
            return "partially_supported", best_cov, best_s  # polarity differs → needs human check
        if best_cov >= 0.7:
            return "supported", best_cov, best_s
        if best_cov >= 0.4:
            return "partially_supported", best_cov, best_s
        return "not_verifiable", best_cov, best_s

    def verify(self, claim: str, evidence: list[str]) -> tuple[str, float, str]:
        label, score, span = self._lexical(claim, evidence)
        if self._nli and span:
            try:
                out = self._nli({"text": span, "text_pair": claim}, truncation=True)
                probs = {o["label"].lower(): o["score"] for o in (out[0] if isinstance(out[0], list) else out)}
                ent = next((v for k, v in probs.items() if k.startswith("entail")), 0.0)
                con = next((v for k, v in probs.items() if k.startswith("contra")), 0.0)
                if con > 0.6:
                    return "contradicted", con, span
                if ent > 0.7 and label != "contradicted":
                    return "supported", ent, span
                if label == "supported":
                    label = "partially_supported"
            except Exception:
                pass
        return label, round(score, 3), span
