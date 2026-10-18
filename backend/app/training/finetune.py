"""Fine-tune a seq2seq summariser (LED by default) on document/summary pairs.

Practices taken from the LED legal-summarisation literature (see research/papers.md):
Adafactor, gradient accumulation, gradient checkpointing, global attention on the first token,
-100 label masking, evaluation on the same held-out split used for every baseline.
NOTE: written to run on a GPU box (Colab/Kaggle/RunPod); it has NOT been executed in the
authoring sandbox (no GPU / no model download). Start with --max-source-len 1024 --limit 50.

  python -m app.training.finetune --data data/splits --out models/led-legal \\
      --model allenai/led-base-16384 --max-source-len 4096 --max-target-len 256 --epochs 3
"""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from pathlib import Path


def build_features(batch, tok, max_src, max_tgt, is_led):
    enc = tok(batch["text"], max_length=max_src, truncation=True)
    lab = tok(text_target=batch["reference_summary"], max_length=max_tgt, truncation=True)
    enc["labels"] = lab["input_ids"]
    if is_led:
        enc["global_attention_mask"] = [[1] + [0] * (len(x) - 1) for x in enc["input_ids"]]
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="allenai/led-base-16384")
    ap.add_argument("--max-source-len", type=int, default=2048); ap.add_argument("--max-target-len", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--limit", type=int); ap.add_argument("--lora", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import numpy as np, torch
    from datasets import load_dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model)
    is_led = model.config.model_type == "led"
    if a.lora:  # parameter-efficient option when GPU memory is tight
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_2_SEQ_LM",
                                                 target_modules=["q_proj", "v_proj"]))
    model.config.use_cache = False

    d = Path(a.data)
    ds = load_dataset("json", data_files={s: str(d / f"{s}.jsonl") for s in ("train", "validation")})
    if a.limit:
        ds = {k: v.select(range(min(a.limit, len(v)))) for k, v in ds.items()}
    cols = ds["train"].column_names
    ds = {k: v.map(lambda b: build_features(b, tok, a.max_source_len, a.max_target_len, is_led),
                   batched=True, remove_columns=cols) for k, v in ds.items()}

    class Collator(DataCollatorForSeq2Seq):  # pads global_attention_mask too; labels padded with -100
        def __call__(self, feats, return_tensors=None):
            g = [f.pop("global_attention_mask", None) for f in feats]
            batch = super().__call__(feats, return_tensors)
            if g[0] is not None:
                L = batch["input_ids"].shape[1]
                batch["global_attention_mask"] = torch.tensor([x + [0] * (L - len(x)) for x in g])
            return batch

    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def metrics(p):
        preds, labels = p
        preds = np.where(preds != -100, preds, tok.pad_token_id); labels = np.where(labels != -100, labels, tok.pad_token_id)
        P, R = tok.batch_decode(preds, skip_special_tokens=True), tok.batch_decode(labels, skip_special_tokens=True)
        out = {k: float(np.mean([sc.score(r, p_)[k].fmeasure for p_, r in zip(P, R)])) for k in ("rouge1", "rouge2", "rougeL")}
        return out

    args = Seq2SeqTrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.accum, learning_rate=a.lr, num_train_epochs=a.epochs, optim="adafactor",
        fp16=torch.cuda.is_available(), gradient_checkpointing=True, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="rougeL", predict_with_generate=True,
        generation_max_length=a.max_target_len, generation_num_beams=2, logging_steps=10, report_to="none", seed=a.seed)
    t0 = time.time()
    tr = Seq2SeqTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
                        data_collator=Collator(tok, model=model, label_pad_token_id=-100), compute_metrics=metrics,
                        processing_class=tok)
    tr.train()
    tr.save_model(a.out); tok.save_pretrained(a.out)
    card = {"base_model": a.model, "hyperparameters": vars(a), "train_seconds": round(time.time() - t0),
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu",
            "dataset_manifest": json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else None,
            "dataset_sha1": hashlib.sha1((d / "train.jsonl").read_bytes()).hexdigest(),
            "final_eval": tr.evaluate(), "limitations": "Fill in: domain, language, failure cases, intended use."}
    (Path(a.out) / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    print("saved", a.out)


if __name__ == "__main__":
    main()
