"""Hierarchical abstractive summarization with LED / BART (lazy-loaded Hugging Face model).

Design notes (see research/papers.md):
- Map-reduce over clause-aware passages so the END of a long contract is never silently truncated.
- LED receives global attention on the first token (Beltagy et al., 2020); this is what the
  JUST-NLP 2025 LED paper also builds on.
- 'hybrid' mode first compresses each passage extractively (BLANCKED, JUST-NLP 2025 found this
  beat small fine-tuned seq2seq models on long legal text), then abstracts.
"""
from __future__ import annotations
import logging
from ..schemas.models import Chunk
from .extractive import candidate_sentences, rank_sentences, select_mmr

log = logging.getLogger(__name__)


class AbstractiveSummarizer:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self._tok = self._model = None
        self.device = device

    # -- lazy load so importing the API never needs torch
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()
        self.max_in = min(getattr(self._tok, "model_max_length", 1024), 4096 if self.device == "cpu" else 16384)

    def token_len(self, text: str) -> int:
        self._load()
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, text: str, max_new_tokens=160, min_new_tokens=30, num_beams=4) -> str:
        self._load()
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=self.max_in).to(self.device)
        kw = {}
        if self._model.config.model_type == "led":
            g = self._torch.zeros_like(enc["input_ids"])
            g[:, 0] = 1  # global attention on <s>
            kw["global_attention_mask"] = g
        with self._torch.no_grad():
            out = self._model.generate(**enc, **kw, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                       num_beams=num_beams, no_repeat_ngram_size=3, length_penalty=1.0,
                                       early_stopping=True)
        return self._tok.decode(out[0], skip_special_tokens=True).strip()


def _group(chunks: list[Chunk], budget_tokens: int, tlen) -> list[list[Chunk]]:
    groups, cur, cur_t = [], [], 0
    for c in chunks:
        t = tlen(c.text)
        if cur and cur_t + t > budget_tokens:
            groups.append(cur)
            cur, cur_t = [], 0
        cur.append(c)
        cur_t += t
    if cur:
        groups.append(cur)
    return groups


def summarize_hierarchical(chunks: list[Chunk], model: AbstractiveSummarizer, hybrid: bool = False):
    """Returns (executive_text, sections, coverage). Raises on model failure (caller falls back)."""
    model._load()
    budget = int(model.max_in * 0.9)
    sections, failures = [], 0
    for grp in _group(chunks, budget, model.token_len):
        if hybrid:  # extractive compression before the neural pass
            ss = candidate_sentences(grp)
            keep = select_mmr(ss, rank_sentences(ss), max(6, len(ss) // 2)) if ss else []
            src = " ".join(ss[i][0] for i in keep) or " ".join(c.text for c in grp)
        else:
            src = "\n".join(c.text for c in grp)
        try:
            summ = model.generate(src, max_new_tokens=140)
        except Exception as exc:  # per-group isolation
            log.warning("group summarization failed: %s", type(exc).__name__)
            failures += 1
            continue
        first, last = grp[0], grp[-1]
        rng = f"§{first.clause_id}–§{last.clause_id}" if first.clause_id and last.clause_id and first is not last \
            else (f"§{first.clause_id}" if first.clause_id else (first.heading or first.chunk_id))
        sections.append({"heading": rng, "summary": summ, "evidence_ids": [c.chunk_id for c in grp]})
    if not sections:
        raise RuntimeError("abstractive summarization produced no output")
    # reduce stage (recursive if the concatenation is itself too long)
    level = [s["summary"] for s in sections]
    while len(level) > 1 and model.token_len(" ".join(level)) > budget:
        merged, buf = [], []
        for s in level:
            if buf and model.token_len(" ".join(buf + [s])) > budget:
                merged.append(model.generate(" ".join(buf), max_new_tokens=140)); buf = []
            buf.append(s)
        if buf:
            merged.append(model.generate(" ".join(buf), max_new_tokens=140))
        level = merged
    executive = level[0] if len(level) == 1 else model.generate(" ".join(level), max_new_tokens=220, min_new_tokens=60)
    coverage = 1 - failures / max(1, failures + len(sections))
    return executive, sections, coverage
