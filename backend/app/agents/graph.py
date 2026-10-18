"""LangGraph workflow: ingest → chunk → summarize → index → analyze → risk → verify ⇄ revise → draft → review.

Deterministic code is used wherever the step is predictable (validation, chunking, schemas, citations);
the only generative steps are the summarizer and (optionally) the LLM in Q&A. The revise loop is bounded
by MAX_VERIFY_RETRIES so cost and latency are capped. A sequential fallback runs the identical nodes
if LangGraph is not installed.
"""
from __future__ import annotations
import time, uuid
from typing import Callable, Optional, TypedDict

from ..config import Settings
from ..ingestion.chunking import chunk_document, split_sentences
from ..ingestion.extract import extract
from ..rag.retriever import HybridRetriever, make_embedder, make_reranker
from ..schemas.models import Chunk, Claim, ClauseHit, KeyFact, Report, RiskFinding, SummaryResult
from ..summarization.pipeline import summarize, summary_sentences
from .clauses import detect_clauses
from .facts import extract_facts
from .risk import detect_risks
from .security import scan_injection
from .verify import Verifier

BAD = ("not_verifiable", "contradicted")


class State(TypedDict, total=False):
    doc_id: str
    filename: str
    data: bytes
    extracted: object
    chunks: list[Chunk]
    summary: SummaryResult
    retriever: HybridRetriever
    facts: list[KeyFact]
    clauses: list[ClauseHit]
    risks: list[RiskFinding]
    claims: list[Claim]
    retries: int
    warnings: list[str]
    unresolved: list[str]
    report: Report


class Pipeline:
    def __init__(self, cfg: Settings, on_stage: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.on_stage = on_stage or (lambda s: None)
        self.verifier = Verifier(cfg.nli_model)

    # ------------------------------------------------------------------ nodes
    def _wrap(self, name, fn):
        def run(state: State):
            self.on_stage(name)
            return fn(state)
        return run

    def n_ingest(self, s: State):
        ex = extract(s["filename"], s["data"], self.cfg.max_upload_mb, self.cfg.max_pages, self.cfg.ocr_enabled)
        warns = list(ex.warnings)
        inj = scan_injection(ex.text)
        if inj:
            warns.append("Document contains text resembling instructions to an AI system "
                         f"({len(inj)} pattern(s)); it was treated as data and ignored.")
        return {"extracted": ex, "warnings": warns, "retries": 0, "unresolved": []}

    def n_chunk(self, s: State):
        chunks = chunk_document(s["extracted"], s["doc_id"], self.cfg.chunk_max_words, self.cfg.chunk_overlap_words)
        if not chunks:
            raise ValueError("No analysable clauses were found in the document.")
        return {"chunks": chunks}

    def n_summarize(self, s: State):
        res = summarize(s["chunks"], self.cfg)
        return {"summary": res, "warnings": s["warnings"] + res.warnings}

    def n_index(self, s: State):
        r = HybridRetriever(s["chunks"], make_embedder(self.cfg.embedding_model if self.cfg.embedding_model != "none" else ""),
                            make_reranker(self.cfg.reranker_model))
        r.save(self.cfg.index_dir / s["doc_id"])
        return {"retriever": r}

    def n_analyze(self, s: State):
        clauses = detect_clauses(s["chunks"])        # also tags chunk.clause_types
        return {"clauses": clauses, "facts": extract_facts(s["chunks"])}

    def n_risk(self, s: State):
        risks = detect_risks(s["chunks"])
        for r in risks:
            r.support = "supported" if r.certainty == "evidence_found" else "not_verifiable"
        return {"risks": risks}

    def _ev_text(self, s: State, ids):
        by = {c.chunk_id: c for c in s["chunks"]}
        return [by[i].text for i in ids if i in by]

    def n_verify(self, s: State):
        claims, n = [], 0
        for text, ids in summary_sentences(s["summary"], s["chunks"]):
            label, score, span = self.verifier.verify(text, self._ev_text(s, ids))
            claims.append(Claim(claim_id=f"k{n:03d}", text=text, evidence_ids=ids[:3], support=label, score=score, best_span=span))
            n += 1
        for f in s["facts"]:  # facts are copied verbatim; check the literal value is in the cited chunk
            ev = " ".join(self._ev_text(s, f.evidence_ids)).lower()
            parts = [p.strip().lower() for p in f.value.split(" and ")] if f.field == "Parties" else [f.value.lower()]
            ok = all(p in ev for p in parts) or f.field == "Auto-renewal"
            claims.append(Claim(claim_id=f"k{n:03d}", text=f"{f.field}: {f.value}", evidence_ids=f.evidence_ids,
                                support="supported" if ok else "not_verifiable", score=1.0 if ok else 0.0))
            n += 1
        return {"claims": claims}

    def route(self, s: State) -> str:
        bad = any(c.support in BAD for c in s["claims"])
        return "revise" if bad and s["retries"] < self.cfg.max_verify_retries else "draft"

    def n_revise(self, s: State):
        """Re-retrieve evidence for unsupported claims using the claim itself as the query."""
        r, k = s["retriever"], self.cfg.retrieval_top_k + 3 * (s["retries"] + 1)
        for c in s["claims"]:
            if c.support not in BAD:
                continue
            hits = r.search(c.text, k)
            ids = [h.chunk_id for h, _ in hits]
            label, score, span = self.verifier.verify(c.text, [h.text for h, _ in hits])
            if score >= c.score or label == "supported":
                c.support, c.score, c.best_span, c.evidence_ids = label, score, span, ids[:3]
        return {"claims": s["claims"], "retries": s["retries"] + 1}

    def n_draft(self, s: State):
        """Assemble the report; drop contradicted sentences, flag unverifiable ones, list unresolved items."""
        summ = s["summary"].model_copy(deep=True)
        removed = {c.text for c in s["claims"] if c.support == "contradicted"}
        unresolved = [f"Contradicted by source (removed from summary): {t}" for t in removed]
        unresolved += [f"Could not verify: {c.text}" for c in s["claims"] if c.support == "not_verifiable"]
        for sec in summ.sections:
            sec.summary = " ".join(x for x in split_sentences(sec.summary) if x not in removed)
        summ.executive_summary = " ".join(x for x in split_sentences(summ.executive_summary) if x not in removed) \
            or "No verifiable summary sentences were produced; see unresolved items."
        for r in s["risks"]:
            if r.certainty == "absence_of_clause":
                unresolved.append(f"Absence finding needs manual confirmation: {r.title}")
        ex = s["extracted"]
        reasons = []
        if any(r.severity == "high" for r in s["risks"]): reasons.append("high-severity potential issue")
        if removed: reasons.append("contradicted summary content removed")
        if unresolved: reasons.append("unresolved items")
        report = Report(
            report_id=uuid.uuid4().hex[:12], doc_id=s["doc_id"], filename=s["filename"], summary=summ,
            key_facts=s["facts"], clauses=s["clauses"], risks=s["risks"], claims=s["claims"],
            unresolved=unresolved, warnings=s["warnings"],
            audit={"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "summarizer_mode": summ.mode,
                   "summarizer_model": summ.model, "embedding": s["retriever"].embedder.name,
                   "verifier": "nli+lexical" if self.verifier._nli else "lexical+numeric",
                   "chunks": len(s["chunks"]), "pages": ex.n_pages, "verify_rounds": s["retries"],
                   "review_required_because": reasons or ["policy: all reports start as pending"]})
        return {"report": report}

    # ------------------------------------------------------------------ build / run
    def _nodes(self):
        return [("ingest", self.n_ingest), ("chunk", self.n_chunk), ("summarize", self.n_summarize),
                ("index", self.n_index), ("analyze", self.n_analyze), ("risk", self.n_risk),
                ("verify", self.n_verify)]

    def build(self):
        from langgraph.graph import END, START, StateGraph
        g = StateGraph(State)
        names = []
        for name, fn in self._nodes():
            g.add_node(name, self._wrap(name, fn))
            names.append(name)
        g.add_node("revise", self._wrap("revise", self.n_revise))
        g.add_node("draft", self._wrap("draft", self.n_draft))
        g.add_edge(START, names[0])
        for a, b in zip(names, names[1:]):
            g.add_edge(a, b)
        g.add_conditional_edges("verify", self.route, {"revise": "revise", "draft": "draft"})
        g.add_conditional_edges("revise", self.route, {"revise": "revise", "draft": "draft"})
        g.add_edge("draft", END)
        return g.compile()

    def run(self, filename: str, data: bytes, doc_id: str) -> State:
        init: State = {"doc_id": doc_id, "filename": filename, "data": data}
        try:
            return self.build().invoke(init, {"recursion_limit": 40})
        except ImportError:  # LangGraph not installed → same nodes, sequential
            s: State = dict(init)
            for name, fn in self._nodes():
                s.update(self._wrap(name, fn)(s))
            while self.route(s) == "revise":
                s.update(self._wrap("revise", self.n_revise)(s))
            s.update(self._wrap("draft", self.n_draft)(s))
            return s
