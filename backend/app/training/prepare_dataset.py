"""Dataset preparation: dedupe, validate, split BY DOCUMENT (never by chunk) and report length stats.

Input JSONL records follow the blueprint schema:
  {"document_id", "document_type", "source", "text", "reference_summary", "clauses": [...]}
Usage:
  python -m app.training.prepare_dataset --in data/raw.jsonl --out data/splits
  python -m app.training.prepare_dataset --billsum --out data/splits_billsum   # pipeline smoke-test corpus only
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def clean_records(records: list[dict]) -> tuple[list[dict], dict]:
    seen, out, stats = set(), [], {"empty": 0, "duplicate": 0, "short_summary": 0}
    for r in records:
        if not r.get("text", "").strip() or not r.get("reference_summary", "").strip():
            stats["empty"] += 1; continue
        if len(r["reference_summary"].split()) < 5:
            stats["short_summary"] += 1; continue
        h = hashlib.sha1(_norm(r["text"]).encode()).hexdigest()
        if h in seen:
            stats["duplicate"] += 1; continue
        seen.add(h)
        r.setdefault("document_id", h[:12])
        out.append(r)
    return out, stats


def split_by_document(records: list[dict], val=0.1, test=0.1, salt="legal-ai-v1") -> dict[str, list[dict]]:
    """Deterministic hash split on document_id (stable as the corpus grows; no chunk-level leakage)."""
    splits = {"train": [], "validation": [], "test": []}
    for r in records:
        u = int(hashlib.sha1(f"{salt}:{r['document_id']}".encode()).hexdigest(), 16) % 10_000 / 10_000
        splits["test" if u < test else "validation" if u < test + val else "train"].append(r)
    return splits


def assert_no_leakage(splits: dict[str, list[dict]]):
    ids = {k: {r["document_id"] for r in v} for k, v in splits.items()}
    for a in ids:
        for b in ids:
            if a < b and ids[a] & ids[b]:
                raise AssertionError(f"document leakage between {a} and {b}")


def length_stats(records, tokenizer=None) -> dict:
    ln = sorted((len(tokenizer(r["text"])["input_ids"]) if tokenizer else len(r["text"].split())) for r in records)
    if not ln:
        return {}
    q = lambda p: ln[min(len(ln) - 1, int(p * len(ln)))]
    return {"n": len(ln), "p50": q(.5), "p90": q(.9), "p99": q(.99), "max": ln[-1],
            "unit": "tokens" if tokenizer else "words"}


def load_billsum(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset  # FiscalNote/billsum
    ds = load_dataset("FiscalNote/billsum", split="train")
    return [{"document_id": f"billsum_{i}", "document_type": "legislation", "source": "FiscalNote/billsum",
             "text": r["text"], "reference_summary": r["summary"]} for i, r in enumerate(ds)][:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp"); ap.add_argument("--out", required=True)
    ap.add_argument("--billsum", action="store_true"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    recs = load_billsum(a.limit) if a.billsum else [json.loads(l) for l in open(a.inp)]
    recs, st = clean_records(recs)
    sp = split_by_document(recs); assert_no_leakage(sp)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k, v in sp.items():
        (out / f"{k}.jsonl").write_text("\n".join(json.dumps(r) for r in v))
    manifest = {"cleaning": st, "sizes": {k: len(v) for k, v in sp.items()}, "length": length_stats(recs)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
