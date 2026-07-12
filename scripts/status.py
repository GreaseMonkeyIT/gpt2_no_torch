"""One-shot training status: progress, perplexity, rate, ETA.

Usage: .venv\\Scripts\\python.exe status.py [out/owt_gpu]
"""

import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRESETS

run = sys.argv[1] if len(sys.argv) > 1 else "out/owt_gpu"
preset = PRESETS[os.path.basename(os.path.normpath(run))]
max_iters = preset["max_iters"]
tok_iter = preset["batch_size"] * preset["grad_accum"] * preset["model"]["block_size"]

with open(os.path.join(run, "log.csv")) as f:
    rows = list(csv.DictReader(f))
last = rows[-1]
it = int(last["iter"])
loss = float(last["train_loss"])

# recent healthy rate (ignore intervals that straddled a laptop nap)
recent = [float(r["tok_per_sec"]) for r in rows[-20:] if float(r["tok_per_sec"]) > 1500]
rate = sum(recent) / len(recent) if recent else 0.0

bar = "#" * int(30 * it / max_iters)
print(f"[{bar:<30}] iter {it:,}/{max_iters:,} ({100 * it / max_iters:.1f}%)")
print(f"tokens seen  {it * tok_iter / 1e6:,.1f}M")
print(f"train loss   {loss:.4f}   (ppl {math.exp(loss):,.1f})")
vals = [(int(r["iter"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
if vals:
    vi, vl = vals[-1]
    print(f"val loss     {vl:.4f}   (ppl {math.exp(vl):,.1f})  @ iter {vi:,}")
    # single eval = 20 random 256-token blocks -> +-0.1 noise; trend = mean of last 3
    if len(vals) >= 3:
        sm = sum(v for _, v in vals[-3:]) / 3
        print(f"val trend    {sm:.4f}   (ppl {math.exp(sm):,.1f})  [mean of last 3 evals]")
if rate:
    eta = (max_iters - it) * tok_iter / rate
    done = time.strftime("%a %H:%M", time.localtime(time.time() + eta))
    print(f"rate         {rate:,.0f} tok/s   ETA {eta / 3600:.1f}h  (~{done})")
