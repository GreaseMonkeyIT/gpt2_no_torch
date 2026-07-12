"""Downstream LM evals, forward-only: WikiText-2 perplexity + LAMBADA accuracy.

Usage (from repo root):
  .venv\\Scripts\\python.exe evals\\eval_lm.py out\\owt_gpu\\ckpt.npz --task wikitext2
  .venv\\Scripts\\python.exe evals\\eval_lm.py out\\owt_gpu\\ckpt.npz --task lambada --limit 500

Honest simplifications (state them in the README):
- wikitext2: non-overlapping block_size windows; a sliding window scores a bit
  better but costs ~block_size/stride more compute
- lambada: correct iff every BPE token of the final word is the argmax under
  teacher forcing (greedy, exact match)
"""

import argparse
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint  # noqa: E402
from engine import tensor as et  # noqa: E402
from engine.tensor import to_numpy  # noqa: E402


def wikitext2_ppl(model, T, enc):
    # Salesforce/ mirror: datasets>=5.0 dropped the legacy bare-"wikitext" script
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = np.array(enc.encode_ordinary(text), dtype=np.int64)
    total_nll, n = 0.0, 0
    for i in tqdm(range(0, len(ids) - T - 1, T), desc="wikitext2"):
        x, y = ids[None, i:i + T], ids[None, i + 1:i + 1 + T]
        _, loss = model(x, y)
        total_nll += float(loss.data) * T
        n += T
    ppl = float(np.exp(total_nll / n))
    print(f"wikitext2: {n:,} tokens, nll {total_nll / n:.4f}, perplexity {ppl:.2f}")


def lambada_acc(model, T, enc, limit):
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    correct, total = 0, 0
    for ex in tqdm(list(ds)[:limit], desc="lambada"):
        ctx, last = ex["text"].rsplit(" ", 1)
        tgt = enc.encode_ordinary(" " + last)
        seq = enc.encode_ordinary(ctx) + tgt
        seq = seq[-(T + 1):]
        logits, _ = model(np.array([seq[:-1]], dtype=np.int64), None)
        pred = to_numpy(logits.data)[0, -len(tgt):, :].argmax(axis=-1)
        correct += bool((pred == np.array(tgt)).all())
        total += 1
    print(f"lambada: {correct}/{total} = {correct / total:.3f} accuracy "
          f"(random baseline ~0; GPT-2 124M reported 45.99%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--task", choices=["wikitext2", "lambada"], required=True)
    ap.add_argument("--limit", type=int, default=500, help="lambada examples")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    args = ap.parse_args()

    if args.device == "gpu":
        # 3GB pool cap, matching the training run (fail loud, don't page)
        os.environ.setdefault("CUPY_GPU_MEMORY_LIMIT", "3221225472")
        et.use_gpu()

    try:
        from model import GPT, GPTConfig
    except ImportError as e:
        sys.exit(f"model.py not ready yet ({e})")

    cfg, arrays, it, _ = checkpoint.load(args.ckpt)
    model = GPT(GPTConfig(**cfg["model"]))
    for p, a in zip(model.parameters(), arrays):
        p.data[...] = et.xp.asarray(a)
    T = cfg["model"]["block_size"]
    enc = tiktoken.get_encoding("gpt2")
    print(f"checkpoint from iter {it}, block_size {T}")

    if args.task == "wikitext2":
        wikitext2_ppl(model, T, enc)
    else:
        lambada_acc(model, T, enc, args.limit)


if __name__ == "__main__":
    main()
