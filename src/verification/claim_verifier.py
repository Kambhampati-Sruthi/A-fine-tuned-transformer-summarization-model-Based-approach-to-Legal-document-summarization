
import re
from typing import Dict, List


US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming"
]


def normalize_tokens(text: str) -> List[str]:
    """Normalize text for lexical comparison, not legal reasoning."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", str(text).lower())


def extract_numbers(text: str) -> set:
    """Extract numeric details that may need explicit verification."""
    return set(re.findall(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*%?"
        r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
        r"thirty|sixty|ninety|hundred|thousand)\b",
        str(text).lower()
    ))


def extract_us_states(text: str) -> set:
    """Extract full US state names; this is a limited conflict screen."""
    found = set()
    for state in US_STATES:
        pattern = r"(?<![A-Za-z])" + re.escape(state) + r"(?![A-Za-z])"
        if re.search(pattern, str(text), flags=re.IGNORECASE):
            found.add(state.lower())
    return found


def lexical_coverage(claim: str, evidence: str) -> float:
    claim_tokens = set(normalize_tokens(claim))
    evidence_tokens = set(normalize_tokens(evidence))
    if not claim_tokens:
        return 0.0
    return round(len(claim_tokens & evidence_tokens) / len(claim_tokens), 3)


def verify_claim(
    claim: str,
    evidence_records: List[Dict],
    min_lexical_coverage: float = 0.35
) -> Dict:
    """
    Evidence triage, not legal truth determination.

    SUPPORTED means only that basic screening checks passed.
    UNRESOLVED is used when screening finds a possible conflict or insufficient
    evidence. All outcomes require human review.
    """
    claim = str(claim).strip()
    if not claim:
        raise ValueError("claim must not be empty")

    usable = []
    for record in evidence_records:
        text = str(record.get("text", record.get("excerpt", ""))).strip()
        if text:
            usable.append({
                **record,
                "text": text,
                "lexical_coverage": lexical_coverage(claim, text)
            })

    if not usable:
        return {
            "claim": claim,
            "status": "UNRESOLVED",
            "reason": "No non-empty source evidence was supplied.",
            "lexical_coverage": 0.0,
            "missing_numeric_details": [],
            "jurisdiction_conflict": False,
            "conflicting_jurisdictions": [],
            "evidence": [],
            "human_review_required": True,
            "is_legal_truth_determination": False
        }

    usable.sort(key=lambda r: r["lexical_coverage"], reverse=True)
    best = usable[0]

    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(best["text"])
    missing_numbers = sorted(claim_numbers - evidence_numbers)

    claim_states = extract_us_states(claim)
    evidence_states = extract_us_states(best["text"])

    # Conservative: only flag when both sides name states and the sets differ.
    state_conflict = bool(
        claim_states and evidence_states and claim_states.isdisjoint(evidence_states)
    )

    if state_conflict:
        status = "UNRESOLVED"
        reason = (
            "Possible jurisdiction conflict: the claim and best-matching "
            "evidence name different US states. Review the complete clause."
        )
    elif claim_numbers and missing_numbers:
        status = "UNRESOLVED"
        reason = (
            "One or more numeric details in the claim were not found in the "
            "best-matching evidence. Retrieve the complete clause and review."
        )
    elif best["lexical_coverage"] < min_lexical_coverage:
        status = "UNRESOLVED"
        reason = (
            "Lexical overlap is below the configured screening threshold; "
            "the retrieved passage may be irrelevant or paraphrased."
        )
    else:
        status = "SUPPORTED"
        reason = (
            "Basic lexical screening passed with no detected numeric or "
            "jurisdiction conflict. This is not proof; human verification is required."
        )

    return {
        "claim": claim,
        "status": status,
        "reason": reason,
        "lexical_coverage": best["lexical_coverage"],
        "missing_numeric_details": missing_numbers,
        "jurisdiction_conflict": state_conflict,
        "conflicting_jurisdictions": (
            sorted(claim_states) + sorted(evidence_states)
            if state_conflict else []
        ),
        "evidence": [{
            "chunk_id": best.get("chunk_id", best.get("evidence_id",
                                                       best.get("id", "UNKNOWN"))),
            "section": best.get("section", "UNKNOWN"),
            "start_offset": best.get("start_offset",
                                     best.get("source_start_char")),
            "end_offset": best.get("end_offset",
                                   best.get("source_end_char")),
            "excerpt": best["text"]
        }],
        "human_review_required": True,
        "is_legal_truth_determination": False
    }
