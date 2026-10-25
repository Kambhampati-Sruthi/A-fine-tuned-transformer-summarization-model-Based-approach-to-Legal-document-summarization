
from typing import Dict, List

from retrieval.evidence_retriever import EvidenceRetriever
from verification import claim_verifier


def build_evidence_retriever(section_records: List[Dict]) -> EvidenceRetriever:
    """
    Build a retriever from section-aware records.

    Expected record fields:
      evidence_id, section, title, source_start_char,
      source_end_char, text

    Original source offsets are retained without modification.
    """
    if not section_records:
        raise ValueError("section_records cannot be empty")

    required = ("text", "source_start_char", "source_end_char")
    for index, record in enumerate(section_records):
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(
                f"Record {index} is missing required fields: {missing}"
            )

    return EvidenceRetriever(section_records)


def retrieve_claim_evidence(
    claim: str,
    retriever: EvidenceRetriever,
    top_k: int = 5
) -> Dict:
    """
    Retrieve passages for a claim, then apply conservative lexical triage.

    Retrieval scores rank passages; they are not support probabilities.
    """
    retrieved = retriever.search(claim, top_k=top_k)

    if not retrieved:
        return {
            "claim": claim,
            "status": "UNRESOLVED",
            "reason": "No evidence passages were retrieved.",
            "evidence": [],
            "human_review_required": True,
            "is_legal_truth_determination": False
        }

    # Adapt retrieved records to the verifier's input interface while
    # preserving the source identifiers, offsets and scores.
    verification_records = []
    for record in retrieved:
        verification_records.append({
            **record,
            "chunk_id": record.get("evidence_id", "UNKNOWN"),
            "start_offset": record.get("source_start_char"),
            "end_offset": record.get("source_end_char"),
        })

    verification = claim_verifier.verify_claim(claim, verification_records)

    # Include ranked candidates so a reviewer can inspect alternatives.
    verification["retrieval_candidates"] = [
        {
            "evidence_id": item.get("evidence_id"),
            "section": item.get("section"),
            "title": item.get("title"),
            "source_start_char": item.get("source_start_char"),
            "source_end_char": item.get("source_end_char"),
            "retrieval_score": item.get("retrieval_score"),
            "bm25_score_normalized": item.get("bm25_score_normalized"),
            "tfidf_similarity": item.get("tfidf_similarity"),
            "excerpt": item.get("text", "")
        }
        for item in retrieved
    ]

    verification["retrieval_score_is_probability"] = False
    return verification
