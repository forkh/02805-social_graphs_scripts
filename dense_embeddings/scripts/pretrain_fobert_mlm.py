"""Continued MLM pretraining of vesteinn/FoBERT on the FPSC transcripts (domain adaptation).

Trains FoBERT with the standard masked-language-modeling objective on all
FPSC speeches (data/speeches.json) and saves the adapted model to
models/fobert-fpsc-mlm/. No vote labels are used, so the downstream stance
evaluation (analyze_rocm.py) stays leakage-free. (Contrastive fine-tuning on
pro/con vote pairs would leak, because the axis is built from those votes.)

Usage (from this project folder):
    .venv/bin/python scripts/pretrain_fobert_mlm.py [--epochs 3] [--batch-size 16]

Then embed/analyze the adapted model (slug "fobert-fpsc" is already in
common.MODELS):
    .venv/bin/python scripts/embed_rocm.py --model fobert-fpsc
    .venv/bin/python scripts/analyze_rocm.py --model fobert-fpsc --axis voters
"""

import argparse
import json
import math
import os
import sys

# Single-GPU training: with 2+ visible GPUs the Trainer wraps the model in
# DataParallel, which uses NCCL — broken in this container. Pin to the XTX.
# Must be set before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import ROOT, SPEECHES_JSON, FOVERT_FPSC_DIR  # noqa: E402

BASE = "vesteinn/FoBERT"


def load_tokenizer(name):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from huggingface_hub import hf_hub_download
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        # e.g. vesteinn/FoBERT under transformers v5: load tokenizer.json
        p = hf_hub_download(name, "tokenizer.json")
        return PreTrainedTokenizerFast(tokenizer_file=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE, help="HF id or local path to start from")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default=FOVERT_FPSC_DIR)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no GPU visible — sync with a GPU extra first "
                 "(uv sync --extra rocm or --extra cuda) and pass the device through "
                 "(--device /dev/kfd --device /dev/dri on ROCm)")
    print(f"torch {torch.__version__} (hip {getattr(torch.version, 'hip', None)}) on "
          f"{torch.cuda.get_device_name(0)}")

    with open(SPEECHES_JSON, encoding="utf-8") as f:
        texts = [s["text"] for s in json.load(f)["speeches"] if s.get("text")]
    print(f"loaded {len(texts)} transcripts from {SPEECHES_JSON}")

    from datasets import Dataset
    from transformers import (AutoModelForMaskedLM, DataCollatorForLanguageModeling,
                              Trainer, TrainingArguments)

    tok = load_tokenizer(args.base)
    vocab = tok.get_vocab()
    if tok.pad_token is None:
        tok.pad_token = "<pad>" if "<pad>" in vocab else tok.eos_token
    if tok.mask_token is None and "<mask>" in vocab:
        # the tokenizer.json fallback path does not register special tokens,
        # but DataCollatorForLanguageModeling requires a mask token
        tok.mask_token = "<mask>"

    ds = Dataset.from_dict({"text": texts})

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    ds = ds.map(tok_fn, batched=True, num_proc=4, remove_columns=["text"])
    ds.set_format("torch")

    # small model (124M): keep fp32 master weights, mixed-precision bf16 compute
    model = AutoModelForMaskedLM.from_pretrained(args.base)
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=args.mlm_prob)
    # transformers v5: overwrite_output_dir and warmup_ratio are gone
    warmup = max(1, int(0.06 * args.epochs * math.ceil(len(ds) / args.batch_size)))
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs, learning_rate=args.lr, warmup_steps=warmup,
        bf16=True, logging_steps=100, save_strategy="no", report_to=[],
        dataloader_num_workers=2, seed=42,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved domain-adapted model to {args.out}")
    print("next: .venv/bin/python scripts/embed_rocm.py --model fobert-fpsc")


if __name__ == "__main__":
    main()
