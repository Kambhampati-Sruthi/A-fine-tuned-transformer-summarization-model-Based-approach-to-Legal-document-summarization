
from typing import Dict, List

from pipeline.claim_evidence_pipeline import (
    build_evidence_retriever,
    retrieve_claim_evidence,
)


def analyze_claims(
    section_records: List[Dict],
    claims: List[str],
    top_k: int = 5,
) -> Dict:
    """
    Retrieve source evidence for each claim and run conservative screening.

    This is an evidence-triage report, not a legal opinion or a guarantee
    that the claims or generated summaries are correct.
    """
    if not section_records:
        raise ValueError("section_records must not be empty")

    if not claims:
        raise ValueError("claims must not be empty")

    retriever = build_evidence_retriever(section_records)
    results = []

    for claim in claims:
        result = retrieve_claim_evidence(
            claim=claim,
            retriever=retriever,
            top_k=top_k,
        )
        results.append(result)

    counts = {
        "SUPPORTED": sum(r.get("status") == "SUPPORTED" for r in results),
        "UNRESOLVED": sum(r.get("status") == "UNRESOLVED" for r in results),
        "CONTRADICTED": sum(r.get("status") == "CONTRADICTED" for r in results),
    }

    return {
        "document_section_count": len(section_records),
        "claim_count": len(results),
        "status_counts": counts,
        "human_review_required": True,
        "is_legal_truth_determination": False,
        "results": results,
    }
