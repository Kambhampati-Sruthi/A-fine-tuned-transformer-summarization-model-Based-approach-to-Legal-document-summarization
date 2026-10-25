
"""Policy-driven agent orchestration for legal document summarization.

This module coordinates existing tools. It does not make legal
determinations, and its screening signals require human review.
"""

from datetime import datetime, timezone


class LegalDocumentAgent:
    """Coordinate summarization, evidence retrieval, and claim screening."""

    def __init__(
        self,
        tokenizer,
        model,
        summarize_fn,
        chunk_fn,
        review_fn,
        claim_analysis_fn,
        max_new_tokens=110,
        max_quality_retries=1,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.summarize_fn = summarize_fn
        self.chunk_fn = chunk_fn
        self.review_fn = review_fn
        self.claim_analysis_fn = claim_analysis_fn
        self.max_new_tokens = max_new_tokens
        self.max_quality_retries = max(0, min(max_quality_retries, 1))

    def _log(self, audit, action, details):
        audit.append({
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details,
        })

    def _summarize_chunk(self, chunk, audit, chunk_number):
        """Summarize one chunk with a bounded quality retry."""
        attempts = 0
        summary = ""
        flags = []

        while attempts <= self.max_quality_retries:
            attempts += 1
            self._log(
                audit,
                "summarize_chunk",
                {"chunk": chunk_number, "attempt": attempts},
            )

            summary = self.summarize_fn(
                chunk,
                self.tokenizer,
                self.model,
                max_new_tokens=self.max_new_tokens,
            )
            flags = self.review_fn(chunk, summary)

            # Retry only for an empty or very short summary.
            # Other flags remain visible for human review.
            retryable = (
                not isinstance(summary, str)
                or not summary.strip()
                or len(summary.split()) < 10
            )

            if not retryable or attempts > self.max_quality_retries:
                break

            self._log(
                audit,
                "quality_retry",
                {
                    "chunk": chunk_number,
                    "reason": "empty_or_short_summary",
                },
            )

        if not isinstance(summary, str):
            summary = str(summary) if summary is not None else ""

        return summary, flags, attempts

    def run(
        self,
        document_text,
        claims=None,
        section_records=None,
        max_words=400,
        overlap_words=60,
    ):
        """Run the controlled document analysis workflow."""
        if not isinstance(document_text, str) or not document_text.strip():
            raise ValueError("Document text must not be empty.")

        audit = []
        claims = claims or []

        self._log(audit, "workflow_started", {
            "document_characters": len(document_text),
            "claims_requested": len(claims),
        })

        # 1. Chunk the document.
        chunks = self.chunk_fn(
            document_text,
            max_words=max_words,
            overlap_words=overlap_words,
        )

        if not chunks:
            raise ValueError("No text chunks were produced.")

        self._log(audit, "chunking_completed", {
            "chunk_count": len(chunks),
        })

        # 2. Summarize each chunk and screen its output.
        chunk_results = []
        summaries = []
        all_flags = []

        for index, chunk in enumerate(chunks, start=1):
            summary, flags, attempts = self._summarize_chunk(
                chunk, audit, index
            )

            summaries.append(summary)
            all_flags.extend(
                {"chunk": index, "flag": flag} for flag in flags
            )

            chunk_results.append({
                "chunk_index": index,
                "source_text": chunk,
                "summary": summary,
                "review_flags": flags,
                "attempts": attempts,
            })

        # This is a concatenation of chunk summaries, not a separately
        # generated hierarchical summary.
        combined_summary = "\n\n".join(
            summary for summary in summaries if summary.strip()
        )

        # 3. Run claim/evidence analysis only when claims are supplied.
        claim_results = []

        if claims:
            if not section_records:
                # Build fallback records with the fields required by
                # build_evidence_retriever(), including source offsets.
                section_records = []
                source_offset = 0

                for i, chunk in enumerate(chunks, start=1):
                    start = document_text.find(chunk, source_offset)

                    if start == -1:
                        # The offsets are approximate if the chunk cannot
                        # be found in the original document.
                        start = source_offset

                    end = start + len(chunk)

                    section_records.append({
                        "evidence_id": f"CHUNK-{i:04d}",
                        "section": f"Chunk {i}",
                        "title": f"Chunk {i}",
                        "source_start_char": start,
                        "source_end_char": end,
                        "text": chunk,
                    })

                    source_offset = end

            self._log(audit, "claim_analysis_started", {
                "claim_count": len(claims),
            })

            claim_report = self.claim_analysis_fn(
                section_records, claims
            )

            # Preserve the complete report when the analysis function
            # returns the dictionary used by contract_analysis.analyze_claims.
            if isinstance(claim_report, dict):
                claim_results = claim_report.get("results", [])
                claim_analysis_report = claim_report
            else:
                # Backward compatibility for a function returning a list.
                claim_results = claim_report
                claim_analysis_report = {
                    "results": claim_results,
                    "human_review_required": True,
                    "is_legal_truth_determination": False,
                }

            self._log(audit, "claim_analysis_completed", {
                "requested_claim_count": len(claims),
                "result_count": len(claim_results),
            })

        else:
            claim_analysis_report = {
                "results": [],
                "human_review_required": True,
                "is_legal_truth_determination": False,
            }

        # 4. Report screening results and retain human-review safeguards.
        self._log(audit, "workflow_completed", {
            "chunks_processed": len(chunk_results),
            "claim_results": len(claim_results),
            "human_review_required": True,
        })

        return {
            "summary": combined_summary,
            "chunk_results": chunk_results,
            "quality_flags": all_flags,
            "claim_results": claim_results,
            "claim_analysis_report": claim_analysis_report,
            "audit_log": audit,
            "metadata": {
                "chunks_processed": len(chunk_results),
                "claims_processed": len(claim_results),
                "human_review_required": True,
                "is_legal_truth_determination": False,
                "summary_method": "concatenated_chunk_summaries",
            },
        }
