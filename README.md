# Legal Document Intelligence & Drafting Assistant

Upload a contract (text PDF / DOCX) → source-linked summary, key facts, clause map, potential-issue review, grounded Q&A, verified report export (PDF/DOCX/MD).
**Decision-support only - not legal advice.** See `research/blueprint_review.md` for scope, evidence and honest limitations.

## Architecture
```
React/Vite/Tailwind ──► FastAPI ──► background job ──► LangGraph workflow
  ingest → chunk → summarize → index → analyze → risk → verify ⇄ revise(≤N) → draft → human review
```
| Stage | Default (offline) | Upgrade (optional) |
|---|---|---|
| Summarise | extractive TF-IDF+MMR, source-linked | `abstractive` / `hybrid` hierarchical LED (or your fine-tuned checkpoint) |
| Retrieve | BM25 + TF-IDF vectors + RRF (+FAISS if installed) | sentence-transformers embeddings, cross-encoder rerank |
| Verify | lexical + numeric/date consistency | MNLI cross-encoder (`NLI_MODEL`) |
| Q&A | extractive, abstains on weak evidence | OpenAI-compatible LLM (`LLM_*`), citations validated |

## Run
```bash
# backend (Python 3.12)
cd backend && python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt            # add requirements-ml.txt for neural parts (install PyTorch first)
cp ../.env.example ../.env
pytest -q                                   # 32 tests
uvicorn app.api.main:app --reload           # http://localhost:8000/docs

# frontend
cd frontend && npm install && npm run dev   # http://localhost:5173 (proxies /api to :8000)
```
## Experiments
```bash
python -m app.training.prepare_dataset --in data/raw.jsonl --out data/splits   # dedupe + document-level split
python -m app.training.finetune --data data/splits --out models/led-legal --max-source-len 1024 --limit 50   # pilot first!
SUMMARIZER_MODE=hybrid MODEL_PATH=models/led-legal uvicorn app.api.main:app
python -m app.evaluation.ablation --n 8 --model-a allenai/led-base-16384 --model-b models/led-legal
```
## Status vs blueprint checklist
Done & tested: upload/extraction (+scanned-PDF handling), baseline summariser, clause/fact/risk detection, hybrid RAG with page/section refs, verification, bounded LangGraph workflow, API, UI (builds), exports, security basics, eval + ablation harness, dataset-split tooling.
**Not done:** actual fine-tuning & real-data evaluation, human factuality review, PostgreSQL, deployment. See `research/blueprint_review.md`.
