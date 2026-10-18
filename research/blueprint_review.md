# Review of `Legal_AI_Project_Blueprint.docx`

Every section was checked against external sources and against what actually runs. Status: OK = kept; CHANGED = improved; ADDED = missing from blueprint.

## Accuracy of the references
| Blueprint claim | Result |
|---|---|
| JUST-NLP 2025 LED paper exists (`2025.justnlp-main.22`) | **OK** - confirmed. **Caveat:** it covers *Indian court judgments*, not contracts (blueprint already says to verify). |
| CUAD = 510 contracts / 41 categories; no paired summaries | **OK** - confirmed. |
| LegalBench is not a summarisation corpus | **OK.** |
| BillSum is legislation, use as pipeline smoke-test | **OK.** |
| LED base checkpoint / Hugging Face / LangGraph / RAGAS / FAISS URLs | **Not re-checked** (could not reach these hosts from the sandbox). Check they resolve and that licences permit your use. |
| `@tailwindcss/vite` + `npm create vite` setup | **OK** - built successfully with Vite 6 + Tailwind 4. |

## Design findings
1. **ADDED - hybrid extractive->abstractive path.** Competition papers (BLANCKED, JUST-NLP 2025) found small fine-tuned seq2seq models alone were weak with ~1k training pairs. Blueprint's E0-E4 plan kept; hybrid mode and an extractive default added so the product works before any fine-tuning succeeds.
2. **CHANGED - hybrid retrieval is default, not optional.** Legal queries hinge on exact terms ("Section 12.2", "60 days"). BM25 + dense + RRF + section-number boost.
3. **CHANGED - chunks are clause-aware and small** (LegalBench-RAG: precision matters), with offsets, page and clause id.
4. **ADDED - numeric/date consistency in verification.** ROUGE/BERTScore cannot detect "30 days" -> "60 days". Verifier flags changed figures as `contradicted`. NLI model pluggable (`NLI_MODEL`).
5. **ADDED - PDF reflow, header/footer stripping, abbreviation-safe sentence splitting.** Found by inspecting output: "Co."/"Inc." broke sentences and hard-wrapped PDF lines fragmented summaries.
6. **ADDED - security/ops items the blueprint lists but doesn't specify:** magic-byte validation, random storage names, API-key + per-owner isolation, retention purge, delete endpoint, prompt-injection scanner + delimiting, schema-validated outputs, no text in logs.
7. **ADDED - human review gate** as an API (`POST /review`) and UI, plus `audit` block (models, verifier, rounds, review reasons).
8. **Gap, unchanged:** a good paired *contract* summary dataset still does not exist in the sources reviewed. Multi-LexSum (litigation) is the closest long legal corpus; building a small reviewed contract set remains the critical research task.
9. **Blueprint's own caution upheld:** an LLM API is not fine-tuning. The LLM adapter is optional and used only for Q&A.

## Traceability: blueprint -> code
| Blueprint section | Implementation |
|---|---|
| 5 Architecture / data flow | `agents/graph.py` (LangGraph), `ingestion/`, `rag/`, `summarization/` |
| 6 Data + fine-tuning | `training/prepare_dataset.py` (dedupe, document-level split, leakage assert), `training/finetune.py` |
| 7 RAG | `rag/retriever.py`, `agents/qa.py` (abstains on weak evidence, drops invented citations) |
| 8 Agents | graph nodes; bounded revise loop (`MAX_VERIFY_RETRIES`); human gate |
| 9 API / UI | `api/main.py` (all listed endpoints + clauses, chunks, review, export, delete); `frontend/` |
| 11 Evaluation | `evaluation/metrics.py`, `evaluation/ablation.py` (A-D) |
| 13 Security | `ingestion/extract.py`, `agents/security.py`, API auth/retention |
| 15 Checklist | see README "Status" |

## What is NOT done (be explicit in any write-up)
- **No fine-tuning has been run** (no GPU, no model downloads in the authoring sandbox). `finetune.py` is complete but unexecuted. Ablation rows A and B are identical until you pass real checkpoints.
- Untested here: sentence-transformers, cross-encoder reranker, NLI verifier, LED/BART generation, OCR, LLM adapter against a live endpoint. All fall back gracefully; fallbacks are tested.
- Frontend compiles and the API is smoke-tested live, but the UI was **not exercised in a browser**.
- Metrics on synthetic data validate the harness only.
- Rule-based clause/risk/fact extraction is heuristic English-only; expect misses on unusual drafting.
- PostgreSQL, Docker and cloud deployment are not included.
