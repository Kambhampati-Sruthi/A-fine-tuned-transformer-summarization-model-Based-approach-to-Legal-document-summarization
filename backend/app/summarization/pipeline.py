from __future__ import annotations
import logging
from ..config import Settings
from ..ingestion.chunking import split_sentences
from ..schemas.models import Chunk, SummaryResult, SummarySection
from .extractive import summarize_extractive

log = logging.getLogger(__name__)
_models: dict[str, object] = {}


def _get_model(name: str):
    from .abstractive import AbstractiveSummarizer
    if name not in _models:
        _models[name] = AbstractiveSummarizer(name)
    return _models[name]


def summarize(chunks: list[Chunk], cfg: Settings, mode: str | None = None) -> SummaryResult:
    mode = mode or cfg.summarizer_mode
    warnings: list[str] = []
    model_id = cfg.model_path or cfg.model_name
    if mode in ("abstractive", "hybrid"):
        try:
            from .abstractive import summarize_hierarchical
            m = _get_model(model_id)
            execu, sections, cov = summarize_hierarchical(chunks, m, hybrid=(mode == "hybrid"))
            return SummaryResult(
                executive_summary=execu, mode=mode, model=model_id, coverage=round(cov, 3), warnings=warnings,
                sections=[SummarySection(**s) for s in sections])
        except Exception as exc:  # missing torch/weights/GPU OOM → degrade, never crash the job
            warnings.append(f"Neural summarizer unavailable ({type(exc).__name__}); used extractive baseline.")
            log.warning("falling back to extractive: %s", exc)
            mode = "extractive"
    execu, sections = summarize_extractive(chunks)
    covered = {cid for _, cid in execu} | {e for s in sections for e in s["evidence_ids"]}
    cov = len(covered & {c.chunk_id for c in chunks}) / max(1, len(chunks))
    return SummaryResult(
        executive_summary=" ".join(s for s, _ in execu), mode="extractive", model="tfidf-mmr",
        coverage=round(min(cov, 1.0), 3), warnings=warnings,
        sections=[SummarySection(**s) for s in sections])


def summary_sentences(res: SummaryResult, chunks: list[Chunk]):
    """(text, evidence_ids) pairs for the verification stage."""
    all_ids = [c.chunk_id for c in chunks]
    out = []
    for sec in res.sections:
        for s in split_sentences(sec.summary):
            out.append((s, sec.evidence_ids or all_ids))
    return out
