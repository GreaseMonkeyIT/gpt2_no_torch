"""Char-level tiny Shakespeare -> data/shakespeare_char/{train,val}.bin + meta.pkl.

Adapted from nanoGPT (karpathy, MIT). Bins are uint16 token ids.
"""

import os
import pickle
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "shakespeare_char")

os.makedirs(OUT, exist_ok=True)
txt_path = os.path.join(OUT, "input.txt")
if not os.path.exists(txt_path):
    urllib.request.urlretrieve(URL, txt_path)

with open(txt_path, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)

n = int(0.9 * len(ids))
ids[:n].tofile(os.path.join(OUT, "train.bin"))
ids[n:].tofile(os.path.join(OUT, "val.bin"))
with open(os.path.join(OUT, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": len(chars), "stoi": stoi,
                 "itos": {i: c for c, i in stoi.items()}}, f)

print(f"vocab_size={len(chars)}  train={n:,} tokens  val={len(ids) - n:,} tokens")
