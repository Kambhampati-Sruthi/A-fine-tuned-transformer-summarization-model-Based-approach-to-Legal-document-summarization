# Research basis

Legend - **[V]** verified this session by web search (title/venue/claims checked against the source page);
**[M]** cited from memory - confirm details before putting in a bibliography.

## Long-document / legal summarisation
- **[M]** Beltagy, Peters, Cohan (2020). *Longformer: The Long-Document Transformer.* arXiv:2004.05150 - sparse local+global attention; LED variant. -> `abstractive.py` (global attention on first token).
- **[V]** Sheik et al. (2025). *Hierarchical Long-Document Summarization using LED for Legal Judgments.* JUST-NLP 2025, ACL Anthology 2025.justnlp-main.22 - multi-level LED summarisation with global attention, Adafactor, gradient accumulation. **Note: task is Indian court judgments (L-SUMM), not contracts.** -> map-reduce design + training recipe.
- **[V]** Parada et al. (2025). *Combining Extractive and Generative Methods for Legal Summarization (BLANCKED).* JUST-NLP 2025, 2025.justnlp-main.17 - small seq2seq models fine-tuned on ~1.2k examples underperformed; a hybrid extractive-abstractive pipeline won. -> **why `hybrid` mode exists and why extractive is the baseline.**
- **[V]** Neelamegam & Nirmala (2025). *Contextors at L-SUMM: Retriever-Driven Multi-Generator Summarization.* 2025.justnlp-main.10 - retriever + BART/Pegasus/LED. -> retrieval-before-generation idea.
- **[V]** D & Madasamy (2025). *SCaLAR_NITK @ L-SUMM.* 2025.justnlp-main.12 - rhetorical-role-aware chunking beats naive token chunking. -> clause-aware (not fixed-size) chunking.
- **[M]** Xiao et al. (2022). *PRIMERA.* ACL 2022, arXiv:2110.08499 - LED pre-trained for multi-document summarisation; candidate checkpoint to test.
- **[M]** Shen et al. (2022). *Multi-LexSum.* NeurIPS D&B, arXiv:2206.10883 - paired multi-granularity legal summaries (litigation).
- **[V, unreviewed]** Nagaraj (2026), SJSU master's project - hierarchical chunking vs truncation vs LED on Multi-LexSum. A thesis, not peer reviewed; treat as a lead.

## Contract understanding & retrieval
- **[V]** Hendrycks et al. (2021). *CUAD.* arXiv:2103.06268 - 510 contracts, 41 clause categories (numbers confirmed). -> `clauses.py` taxonomy.
- **[V]** Pipitone et al. (2024). *LegalBench-RAG.* arXiv:2408.10343 - 6,858 expert-annotated query-answer pairs with exact spans; argues for **precise small snippets** over large chunks. -> small clause-bounded chunks, offsets on every hit.
- **[M]** Guha et al. (2023). *LegalBench.* arXiv:2308.11462.
- **[M]** Koreeda & Manning (2021). *ContractNLI.* arXiv:2110.01799 -> training data for an NLI verifier.
- **[M]** Chalkidis et al. (2022). *LexGLUE.* arXiv:2110.00976; Chalkidis et al. (2020). *LEGAL-BERT.* arXiv:2010.02559 -> upgrade path for the rule-based clause classifier.
- **[M]** Lewis et al. (2020). *Retrieval-Augmented Generation.* arXiv:2005.11401.
- **[M]** Cormack, Clarke, Buettcher (2009). *Reciprocal Rank Fusion.* SIGIR -> `retriever.py` fusion of BM25 + dense.
- **[M]** Liu et al. (2023). *Lost in the Middle.* arXiv:2307.03172 -> keep top-k small; most relevant passage first.

## Faithfulness, evaluation, safety
- **[M]** Laban et al. (2022). *SummaC: Re-visiting NLI-based Inconsistency Detection in Summarization.* TACL, arXiv:2111.09525 -> sentence-level verification in `verify.py`.
- **[M]** Magesh et al. (2024). *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools.* arXiv:2405.20362 -> commercial RAG legal tools still hallucinate; motivates verification, uncertainty labels, human review.
- **[M]** Es et al. (2023). *RAGAS.* arXiv:2309.15217; Zhang et al. (2020). *BERTScore.* arXiv:1904.09675; Lin (2004). *ROUGE.*
- **[M]** Greshake et al. (2023). *Not what you've signed up for: Indirect Prompt Injection.* arXiv:2302.12173 -> `security.py` (contract text = untrusted data).
- **[M]** Hu et al. (2021). *LoRA.* arXiv:2106.09685 -> `--lora` option.
