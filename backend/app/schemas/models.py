"""Pydantic contracts shared by every pipeline stage (validated before rendering)."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

Support = Literal["supported", "partially_supported", "contradicted", "not_verifiable"]
Severity = Literal["info", "low", "medium", "high"]
AnswerType = Literal["explicit", "inferred", "ambiguous", "absent"]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    clause_id: Optional[str] = None      # e.g. "12.2"
    heading: Optional[str] = None
    text: str
    page: Optional[int] = None           # 1-based; None for DOCX (no fixed pagination)
    page_end: Optional[int] = None
    start: int                           # char offsets into cleaned document text
    end: int
    clause_types: list[str] = Field(default_factory=list)

    def ref(self) -> str:
        parts = []
        if self.clause_id:
            parts.append(f"§{self.clause_id}")
        elif self.heading:
            parts.append(self.heading[:40])
        if self.page:
            parts.append(f"p.{self.page}")
        return ", ".join(parts) or self.chunk_id


class Evidence(BaseModel):
    chunk_id: str
    ref: str
    snippet: str
    score: float = 0.0


class SummarySection(BaseModel):
    heading: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)


class SummaryResult(BaseModel):
    executive_summary: str
    sections: list[SummarySection]
    mode: str
    model: str
    coverage: float = Field(ge=0, le=1, description="fraction of clauses represented")
    warnings: list[str] = Field(default_factory=list)


class KeyFact(BaseModel):
    field: str
    value: str
    evidence_ids: list[str] = Field(default_factory=list)


class ClauseHit(BaseModel):
    clause_type: str
    chunk_id: str
    ref: str
    snippet: str
    confidence: float


class RiskFinding(BaseModel):
    rule_id: str
    title: str
    severity: Severity
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)
    certainty: Literal["evidence_found", "absence_of_clause", "needs_review"] = "evidence_found"
    support: Optional[Support] = None
    disclaimer: str = "Potential issue for human review - not a legal conclusion."


class Claim(BaseModel):
    claim_id: str
    text: str
    evidence_ids: list[str]
    support: Support = "not_verifiable"
    score: float = 0.0
    best_span: str = ""


class QAAnswer(BaseModel):
    question: str
    answer: str
    answer_type: AnswerType
    evidence: list[Evidence]
    support: Support = "not_verifiable"
    note: str = ""


class Report(BaseModel):
    report_id: str
    doc_id: str
    filename: str
    summary: SummaryResult
    key_facts: list[KeyFact]
    clauses: list[ClauseHit]
    risks: list[RiskFinding]
    claims: list[Claim]
    unresolved: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    audit: dict = Field(default_factory=dict)
    review_status: Literal["pending", "approved", "edited", "rejected"] = "pending"
    reviewer_notes: str = ""
    disclaimer: str = (
        "Decision-support and drafting aid only. Not legal advice. "
        "A qualified professional must review before any material decision."
    )
