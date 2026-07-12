"""Generate text from a checkpoint.

Usage: .venv\\Scripts\\python.exe sample.py out/shakespeare_char/ckpt.npz
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint
from engine import tensor as et

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--prompt", default="\n")
ap.add_argument("--num-tokens", type=int, default=300)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=200)
ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
args = ap.parse_args()

if args.device == "gpu":
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

meta_path = os.path.join(cfg["data_dir"], "meta.pkl")
if os.path.exists(meta_path):  # char-level
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    encode = lambda s: [meta["stoi"][c] for c in s]
    decode = lambda ids: "".join(meta["itos"][i] for i in ids)
else:  # GPT-2 BPE
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode, decode = enc.encode_ordinary, enc.decode

idx = np.array([encode(args.prompt)], dtype=np.int64)
out = model.generate(idx, args.num_tokens, temperature=args.temperature,
                     top_k=args.top_k)
print(decode(np.asarray(out)[0].tolist()))
