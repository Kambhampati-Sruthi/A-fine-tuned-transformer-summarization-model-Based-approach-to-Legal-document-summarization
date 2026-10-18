"""Transparent, rule-based 'potential issue' detection. Each rule has an explicit definition,
requires evidence (or states the absence of a clause), and never claims a legal conclusion."""
from __future__ import annotations
import re
from ..schemas.models import Chunk, RiskFinding

I = re.I


def _chunks_with(chunks, label):
    return [c for c in chunks if label in c.clause_types]


def _first(chunks, rx):
    return [c for c in chunks if re.search(rx, c.text, I)]


def detect_risks(chunks: list[Chunk]) -> list[RiskFinding]:
    R: list[RiskFinding] = []

    def add(rule_id, title, sev, why, ev, certainty="evidence_found"):
        R.append(RiskFinding(rule_id=rule_id, title=title, severity=sev, rationale=why,
                             evidence_ids=[c.chunk_id for c in ev][:3], certainty=certainty))

    # --- presence-based rules -------------------------------------------------------------
    ev = _first(chunks, r"automatically\s+renew|auto-?renew|renew\w*\s+automatically")
    if ev:
        short = _first(ev, r"\b(?:30|thirty|60|sixty|90|ninety)\b[^.]{0,40}(?:days|notice)")
        add("R-RENEW", "Automatic renewal", "medium" if short else "low",
            "The agreement appears to renew automatically. Check the opt-out notice window and "
            "any renewal price changes so a renewal date is not missed.", ev)

    ev = _first(chunks, r"terminate[^.]{0,100}(?:for any reason|without cause|for convenience|at (?:its|their|any) (?:sole )?(?:discretion|time))")
    if ev:
        add("R-TERM-CONV", "Termination without cause", "medium",
            "A party may be able to end the agreement without stating a reason. Confirm whether this right is "
            "mutual and what notice or fees apply.", ev)

    ev = _first(chunks, r"sole (?:and absolute )?discretion")
    if ev:
        add("R-DISCRETION", "Discretionary / unilateral rights", "medium",
            "Language giving one party 'sole discretion' may allow unilateral decisions. Review who holds the right.", ev)

    ev = [c for c in _chunks_with(chunks, "Indemnification")]
    if ev:
        capped = any(re.search(r"cap|not exceed|limited to|aggregate", c.text, I) for c in _chunks_with(chunks, "Limitation of Liability") + ev)
        add("R-INDEM", "Indemnification obligation", "medium" if capped else "high",
            "An indemnity is present" + ("." if capped else " and no liability cap was detected nearby; exposure may be uncapped."), ev)

    ev = _first(chunks, r"non-?compet\w+|shall not[^.]{0,60}(?:compete|engage)")
    if ev:
        add("R-NONCOMPETE", "Non-compete / restriction on activity", "medium",
            "A restriction on competing or engaging in certain business appears. Check scope, geography and duration.", ev)

    ev = _first(chunks, r"exclusive(?:ly)?\b")
    if ev:
        add("R-EXCL", "Exclusivity", "low", "Exclusivity language may limit working with others. Confirm its scope and duration.", ev)

    ev = _first(chunks, r"(?:perpetual|indefinite(?:ly)?|in perpetuity)")
    if ev:
        add("R-PERPETUAL", "Perpetual / indefinite obligation", "medium",
            "An obligation or licence appears to continue indefinitely. Check whether it survives termination as intended.", ev)

    ev = _first(chunks, r"(?:may|reserves the right to)\s+(?:amend|modify|change|update)[^.]{0,80}(?:at any time|from time to time|without)")
    if ev:
        add("R-UNILATERAL-AMEND", "Unilateral amendment right", "high",
            "One party may be able to change terms without the other's agreement.", ev)

    ev = [c for c in _first(chunks, r"assign\w*[^.]{0,100}without[^.]{0,40}consent") ]
    if ev:
        add("R-ASSIGN", "Assignment / change-of-control restriction", "low",
            "Assignment appears to require consent (or is restricted). Check whether the restriction applies to both parties.", ev)

    for c in _chunks_with(chunks, "Late Fees / Interest"):
        m = re.search(r"(\d+(?:\.\d+)?)\s?%\s*per\s*month", c.text, I)
        if m and float(m.group(1)) > 1.5:
            add("R-LATEFEE", "High late-payment interest", "medium",
                f"A late charge of {m.group(1)}% per month was found, which is a high annualised rate.", [c])

    ev = _first(chunks, r"arbitrat\w+")
    if ev:
        add("R-ARB", "Binding arbitration / forum clause", "low",
            "Disputes may be resolved by arbitration or in a specified forum. Check seat, rules and cost allocation.", ev)

    ev = _first(chunks, r"(?:non-?refundable|no refunds?)")
    if ev:
        add("R-NOREFUND", "Non-refundable payments", "low", "Some payments appear to be non-refundable.", ev)

    # --- absence-based rules (explicitly flagged as absence, needing manual confirmation) ---
    have = {t for c in chunks for t in c.clause_types}
    for label, rid, sev, why in [
        ("Governing Law", "R-NO-LAW", "medium", "No governing-law clause was detected."),
        ("Termination", "R-NO-TERM", "medium", "No termination clause was detected."),
        ("Limitation of Liability", "R-NO-LOL", "medium", "No limitation-of-liability clause was detected."),
        ("Confidentiality", "R-NO-CONF", "low", "No confidentiality clause was detected."),
    ]:
        if label not in have:
            add(rid, f"Missing: {label}", sev,
                why + " This is based on automated pattern matching; confirm manually, as the clause may be worded unusually.",
                [], "absence_of_clause")
    return R
