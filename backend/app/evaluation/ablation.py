"""Ablation runner (blueprint §11): A pretrained · B fine-tuned · C +RAG QA · D +verification.
Run:  python -m app.evaluation.ablation --n 8
Without model weights it compares the extractive baseline against the full verified pipeline on synthetic
data so the harness itself is testable; point --model-a/--model-b at real checkpoints for real ablations."""
from __future__ import annotations
import argparse, json, time
from ..agents.graph import Pipeline
from ..agents.qa import answer_question
from ..agents.verify import Verifier
from ..config import Settings
from ..ingestion.chunking import chunk_document
from ..ingestion.extract import extract
from ..rag.retriever import HybridRetriever, make_embedder
from . import metrics as M
from .synthetic import make_contract, to_docx_bytes


def run(n: int, cfg: Settings, model_a: str | None = None, model_b: str | None = None):
    rows = []
    for seed in range(n):
        ex = make_contract(seed)
        data = to_docx_bytes(ex["text"])
        # A / B: summarizer-only variants (pretrained vs fine-tuned if paths supplied, else extractive)
        variants = {"A_baseline": (cfg.model_copy(update={"summarizer_mode": "abstractive" if model_a else "extractive",
                                                           "model_name": model_a or cfg.model_name})),
                    "B_finetuned": (cfg.model_copy(update={"summarizer_mode": "abstractive" if model_b else "extractive",
                                                           "model_path": model_b or ""}))}
        for name, c in variants.items():
            t = time.time()
            out = Pipeline(c.model_copy(update={"max_verify_retries": 0})).run("x.docx", data, f"ab{seed}")
            rep = out["report"]
            rows.append({"seed": seed, "system": name, "latency_s": round(time.time() - t, 2),
                         **M.rouge_all(rep.summary.executive_summary, ex["reference_summary"]),
                         **M.factuality_from_claims(rep.claims)})
        # C / D: add retrieval QA, then verification-aware reporting
        out = Pipeline(cfg).run("x.docx", data, f"cd{seed}")
        r = out["retriever"]
        ranked = [[h.clause_id for h, _ in r.search(q, 5)] for q, _ in ex["qa"]]
        gold = [g for _, g in ex["qa"]]
        ver = Verifier(cfg.nli_model)
        qa = [answer_question(q, r, cfg, ver) for q, _ in ex["qa"]]
        rows.append({"seed": seed, "system": "C_rag_qa", "recall@1": M.recall_at_k(ranked, gold, 1),
                     "recall@3": M.recall_at_k(ranked, gold, 3), "mrr": M.mrr(ranked, gold),
                     "qa_supported": sum(a.support == "supported" for a in qa) / len(qa)})
        rep = out["report"]
        rows.append({"seed": seed, "system": "D_verified", **M.rouge_all(rep.summary.executive_summary, ex["reference_summary"]),
                     **M.factuality_from_claims(rep.claims), "unresolved": len(rep.unresolved)})
    return rows


def summarize(rows):
    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and k != "seed":
                agg[r["system"]][k].append(v)
    return {s: {k: round(sum(v) / len(v), 3) for k, v in d.items()} for s, d in agg.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--model-a"); ap.add_argument("--model-b")
    ap.add_argument("--out", default="../outputs/ablation.json")
    a = ap.parse_args()
    cfg = Settings(embedding_model="none", database_url="sqlite:///:memory:")
    res = summarize(run(a.n, cfg, a.model_a, a.model_b))
    print(json.dumps(res, indent=2))
    open(a.out, "w").write(json.dumps(res, indent=2))
