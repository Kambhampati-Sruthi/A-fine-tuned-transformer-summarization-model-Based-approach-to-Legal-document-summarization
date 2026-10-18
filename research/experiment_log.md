# Experiment log

| ID | System | Data | Status | Notes |
|----|--------|------|--------|-------|
| E0 | Extractive TF-IDF/MMR (`summarizer_mode=extractive`) | synthetic (harness check only) | harness run | ROUGE-1 ~0.24 vs synthetic reference; **not a real result** |
| E1 | Pretrained LED (`abstractive`) | - | TODO (needs weights/GPU) | `python -m app.evaluation.ablation --model-a allenai/led-base-16384` |
| E2 | Fine-tuned LED | - | TODO | `app.training.finetune`, then `--model-b models/led-legal` |
| E3 | Hierarchical LED / hybrid | - | TODO | `SUMMARIZER_MODE=hybrid` |
| E4 | E2/E3 + verification | - | TODO | compare `pct_contradicted` and unsupported-claim rate |

Rules: same held-out documents for every row; never tune on the test split; record seed, hardware, dataset sha1 (written to `model_card.json`).
Retrieval (synthetic, 8 docs x 5 questions): recall@3 = 1.0, MRR = 0.96 - an easy templated corpus; re-measure on CUAD / LegalBench-RAG.
