"""Stream an OpenWebText subset -> data/openwebtext/{train,val}.bin (uint16 GPT-2 BPE).

Full OWT is ~9B tokens / 54GB raw — far beyond weekend compute, so we stream the
first --max-tokens (default 100M) instead of downloading everything. Every 200th
doc goes to val (~0.5%). <|endoftext|> (50256) separates documents.

Usage: python data/prepare_openwebtext.py [--max-tokens 100000000]
"""

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--max-tokens", type=int, default=100_000_000)
args = parser.parse_args()

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "openwebtext")
os.makedirs(OUT, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

counts = {"train": 0, "val": 0}
files = {s: open(os.path.join(OUT, f"{s}.bin"), "wb") for s in counts}
pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True)

for i, doc in enumerate(ds):
    ids = enc.encode_ordinary(doc["text"]) + [enc.eot_token]
    split = "val" if i % 200 == 0 else "train"
    files[split].write(np.array(ids, dtype=np.uint16).tobytes())
    counts[split] += len(ids)
    pbar.update(len(ids))
    if counts["train"] + counts["val"] >= args.max_tokens:
        break

pbar.close()
for f in files.values():
    f.close()
print(f"done: train={counts['train']:,} tokens  val={counts['val']:,} tokens")
