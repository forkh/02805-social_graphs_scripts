"""Embed the speeches needed for contested ballots on the GPU (ROCm/HIP or CUDA).

Port of the original models/embedding_stance_rocm/scripts/embed_rocm.py. A ROCm
build of PyTorch exposes HIP through the same torch.cuda API, so device="cuda"
works unchanged on both backends; this script just detects the device, prints
the HIP/CUDA version, applies per-model input prefixes (e5) / task adapters,
per-model dtypes (bf16 for big models) and per-model default batch
sizes.

Usage (from this project folder):
    .venv/bin/python scripts/embed_rocm.py --model minilm [--batch-size 256]
    .venv/bin/python scripts/embed_rocm.py --model minilm

Writes data/speech_embeddings_<slug>.npz (ids + float32 matrix, L2-normalized)
and prints timing + GPU memory. Re-runs reuse the cache and only embed what is
missing.

Models (see common.MODELS): the CUDA models (validation), the bigger encoders,
and the run_queue.sh follow-ups (incl. the locally pretrained fobert-fpsc path).
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MODELS, PLAIN_ENCODERS, PREFIXES, BF16_MODELS, TASKS,
                    DEFAULT_BATCH, DATA_OUT, emb_cache_path, load_speeches_by_case,
                    load_votes_ballots, needed_speech_ids, gpu_info)


class MeanPoolEncoder:
    """Plain HF encoder (BERT/RoBERTa/XLM-R) or local model path, wrapped for
    .encode(), with mean pooling + optional L2 norm. Tokenization is
    hard-truncated at 512 tokens — several Nordic checkpoints ship tokenizer
    configs with an unbounded model_max_length, and >512-token inputs overflow
    the position embeddings."""

    MAX_LEN = 512

    def __init__(self, name, device, dtype=None):
        import torch
        from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
        from huggingface_hub import hf_hub_download
        self.device = device
        self.torch = torch
        try:
            self.tok = AutoTokenizer.from_pretrained(name)
        except Exception:
            # e.g. vesteinn/FoBERT under transformers v5: load tokenizer.json
            p = hf_hub_download(name, "tokenizer.json")
            self.tok = PreTrainedTokenizerFast(tokenizer_file=p)
        if self.tok.pad_token is None:
            vocab = self.tok.get_vocab()
            self.tok.pad_token = "<pad>" if "<pad>" in vocab else self.tok.eos_token
        kw = {"torch_dtype": dtype} if dtype is not None else {}
        self.model = AutoModel.from_pretrained(name, **kw).to(device).eval()

    def encode(self, texts, batch_size=256, normalize_embeddings=True,
               show_progress_bar=False, convert_to_numpy=True):
        import numpy as np
        from tqdm import tqdm
        torch = self.torch
        outs = []
        it = range(0, len(texts), batch_size)
        if show_progress_bar:
            it = tqdm(it, desc="Batches")
        with torch.no_grad():
            for i in it:
                batch = texts[i:i + batch_size]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.MAX_LEN, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hidden = self.model(**enc).last_hidden_state      # b x t x d
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                if normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                outs.append(emb.cpu().numpy())
        return np.vstack(outs) if convert_to_numpy else torch.cat(outs)


def load_model(name, device):
    from sentence_transformers import SentenceTransformer
    dtype = torch.bfloat16 if name in BF16_MODELS else None
    if name in PLAIN_ENCODERS or os.path.isdir(name):
        return MeanPoolEncoder(name, device, dtype=dtype)
    kw = {"model_kwargs": {"torch_dtype": dtype}} if dtype is not None else {}
    # trust_remote_code: some repos ship custom ST modules;
    # ST >= 6 refuses them without this flag.
    return SentenceTransformer(name, device=device, trust_remote_code=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: per-model DEFAULT_BATCH from common.py")
    args = ap.parse_args()

    info = gpu_info()
    if info is None:
        sys.exit("no GPU visible — sync with a GPU extra first "
                 "(uv sync --extra rocm or --extra cuda) and pass the device through "
                 "(--device /dev/kfd --device /dev/dri on ROCm)")
    device = "cuda"  # ROCm torch exposes HIP via the CUDA API
    print(f"torch {torch.__version__} (hip {getattr(torch.version, 'hip', None)}) on "
          f"{info['name']} ({info['mem_gb']:.1f} GB)")

    slug = args.model
    name = MODELS[slug]
    batch_size = args.batch_size or DEFAULT_BATCH.get(slug, 128)
    prefix = PREFIXES.get(name, "")
    task = TASKS.get(name)
    cache = emb_cache_path(slug)

    by_case, n_speeches = load_speeches_by_case()
    votes, ballots = load_votes_ballots()
    needed = needed_speech_ids(by_case, votes, ballots)
    print(f"speeches total={n_speeches}, case-linked={sum(len(v) for v in by_case.values())}, "
          f"needed for contested ballots={len(needed)}")

    emb = {}
    if os.path.exists(cache):
        z = np.load(cache)
        mat = z["matrix"]  # decompress ONCE
        emb = {i: mat[k] for k, i in enumerate(z["ids"])}
        del z
    missing = {i: t for i, t in needed.items() if i not in emb}
    print(f"cache {cache}: {len(emb)} cached, {len(missing)} to embed")
    if not missing:
        return

    t0 = time.time()
    model = load_model(name, device)
    ids = sorted(missing)
    texts = [prefix + missing[i] for i in ids]
    enc_kw = {"task": task} if (task and name not in PLAIN_ENCODERS and not os.path.isdir(name)) else {}
    mat = model.encode(texts, batch_size=batch_size,
                       normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True, **enc_kw)
    dt = time.time() - t0
    emb.update({i: mat[k] for k, i in enumerate(ids)})
    print(f"embedded {len(ids)} speeches in {dt:.1f}s "
          f"({len(ids) / dt:.0f} sp/s, batch={batch_size}) on {device} ({info['name']}); "
          f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

    all_ids = sorted(emb)
    os.makedirs(DATA_OUT, exist_ok=True)
    np.savez_compressed(cache, ids=np.array(all_ids),
                        matrix=np.vstack([emb[i] for i in all_ids]).astype(np.float32))
    print(f"wrote {cache} ({len(all_ids)} x {mat.shape[1]})")


if __name__ == "__main__":
    main()
