"""Full-val perplexity: the headline number the 20-block training estimator only
approximated. Scores every non-overlapping block_size window of val.bin once
(forward-only, CPU) and reports the token-weighted mean NLL + perplexity.

Usage (from repo root):
  .venv\\Scripts\\python.exe evals\\full_val.py out\\owt_gpu\\ckpt.npz

Honest simplification (state it in the README): non-overlapping windows, same as
eval_lm.py's wikitext2 — a sliding window scores marginally better at ~T/stride
more compute. Batched only for speed; the number is identical to a loop of ones.
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint  # noqa: E402
from engine import tensor as et  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", default=None,
                    help="val.bin path (default: <data_dir>/val.bin from ckpt)")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--batch", type=int, default=16, help="windows per forward")
    args = ap.parse_args()

    if args.device == "gpu":
        # same 3GB pool cap as the training run: overflow fails loudly (OOM)
        # instead of paging over PCIe. use_gpu() must precede any Tensor build.
        os.environ.setdefault("CUPY_GPU_MEMORY_LIMIT", "3221225472")
        et.use_gpu()

    try:
        from model import GPT, GPTConfig
    except ImportError as e:
        sys.exit(f"model.py not ready yet ({e})")

    cfg, arrays, it, _ = checkpoint.load(args.ckpt)
    model = GPT(GPTConfig(**cfg["model"]))
    for p, a in zip(model.parameters(), arrays):
        p.data[...] = et.xp.asarray(a)      # -> device (cupy) when gpu, else numpy
    T = cfg["model"]["block_size"]

    data_path = args.data or os.path.join(cfg["data_dir"], "val.bin")
    ids = np.fromfile(data_path, dtype=np.uint16).astype(np.int64)
    print(f"checkpoint iter {it}, block_size {T}, val tokens {len(ids):,}")

    starts = list(range(0, len(ids) - T - 1, T))     # non-overlapping windows
    total_nll, n = 0.0, 0
    for b in tqdm(range(0, len(starts), args.batch), desc="full_val"):
        js = starts[b:b + args.batch]
        x = np.stack([ids[j:j + T] for j in js])
        y = np.stack([ids[j + 1:j + 1 + T] for j in js])
        _, loss = model(x, y)                          # mean NLL over this batch
        total_nll += float(loss.data) * len(js) * T    # un-mean -> token sum
        n += len(js) * T
    nll = total_nll / n
    print(f"full val: {n:,} tokens, nll {nll:.4f}, perplexity {np.exp(nll):.2f}")


if __name__ == "__main__":
    main()
