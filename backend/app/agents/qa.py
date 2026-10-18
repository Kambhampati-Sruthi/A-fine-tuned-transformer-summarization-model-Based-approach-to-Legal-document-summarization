"""Grounded contract Q&A: hybrid retrieval → (optional LLM | extractive) answer → verification.
Refuses to answer when evidence is weak, and always returns evidence references."""
from __future__ import annotations
from ..config import Settings
from ..ingestion.chunking import split_sentences
from ..rag.retriever import HybridRetriever, tokenize
from ..schemas.models import Evidence, QAAnswer
from .llm import LLM
from .security import SYSTEM_PROMPT, wrap_untrusted
from .verify import Verifier

MIN_CONF = 0.25


def answer_question(question: str, retriever: HybridRetriever, cfg: Settings,
                    verifier: Verifier, llm: LLM | None = None) -> QAAnswer:
    hits = retriever.search(question, cfg.retrieval_top_k)
    evidence = [Evidence(chunk_id=c.chunk_id, ref=c.ref(), snippet=c.text[:400], score=round(s, 4)) for c, s in hits]
    qtoks = set(tokenize(question))
    conf = max((retriever.lexical_confidence(question, c.text) for c, _ in hits), default=0.0)
    if not hits or conf < MIN_CONF:
        return QAAnswer(question=question, answer="The document does not appear to address this question "
                        "(no sufficiently relevant passage was retrieved).", answer_type="absent",
                        evidence=evidence[:2], support="not_verifiable", note="Low retrieval confidence.")

    ans, atype, used = None, None, None
    if llm and llm.enabled:
        ctx = "\n\n".join(f"[{c.chunk_id}] ({c.ref()})\n{c.text}" for c, _ in hits)
        res = llm.complete_json(SYSTEM_PROMPT, f"Question: {question}\n\nEvidence:\n{wrap_untrusted(ctx)}")
        if res and isinstance(res.get("answer"), str):
            allowed = {c.chunk_id for c, _ in hits}
            used = [e for e in res.get("evidence_ids", []) if e in allowed]   # drop invented citations
            ans, atype = res["answer"], res.get("answer_type", "inferred")
            if atype not in ("explicit", "inferred", "ambiguous", "absent") or not used:
                ans = None  # unusable → fall back
    if ans is None:  # extractive: best sentences from top passages
        scored = []
        for c, _ in hits[:3]:
            for s in split_sentences(c.text):
                cov = len(qtoks & set(tokenize(s))) / (len(qtoks) or 1)
                scored.append((cov, s, c.chunk_id))
        scored.sort(key=lambda x: -x[0])
        top = scored[:2]
        ans = " ".join(s for _, s, _ in top) or hits[0][0].text[:300]
        used = list(dict.fromkeys(cid for _, _, cid in top)) or [hits[0][0].chunk_id]
        best = top[0][0] if top else 0
        atype = "explicit" if best >= 0.6 else "inferred" if best >= 0.35 else "ambiguous"

    ev_texts = [retriever.by_id[i].text for i in used if i in retriever.by_id]
    # verify sentence-by-sentence (SummaC-style) and keep the weakest label: one bad sentence taints the answer
    order = ["contradicted", "not_verifiable", "partially_supported", "supported"]
    labels = [verifier.verify(s, ev_texts)[0] for s in (split_sentences(ans) or [ans])]
    label = min(labels, key=order.index)
    ev_sel = [e for e in evidence if e.chunk_id in used] or evidence[:2]
    note = "" if label == "supported" else "Answer could not be fully verified against the cited passage; check the source."
    return QAAnswer(question=question, answer=ans, answer_type=atype, evidence=ev_sel, support=label, note=note)
