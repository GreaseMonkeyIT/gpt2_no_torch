"""End-to-end model checks on a tiny config, in float64.

1. init loss ~= ln(vocab_size): an untrained model should be uniformly unsure.
2. full-model gradient spot-check: perturb random entries of EVERY parameter
   and compare the loss's numeric derivative against the engine's backward.

Run: .venv\\Scripts\\python.exe tests\\test_model.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GPT, GPTConfig  # noqa: E402

rng = np.random.default_rng(3)
cfg = GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=8, vocab_size=23)
model = GPT(cfg)
for p in model.parameters():          # float64 for finite differences
    p.data = p.data.astype(np.float64)

B, T = 2, 8
idx = rng.integers(0, cfg.vocab_size, (B, T))
tgt = rng.integers(0, cfg.vocab_size, (B, T))

# -- 1. init loss sanity ------------------------------------------------------
_, loss = model(idx, tgt)
expect = np.log(cfg.vocab_size)
print(f"init loss {float(loss.data):.4f} vs ln(V)={expect:.4f}")
assert abs(float(loss.data) - expect) < 0.5, "init loss far from uniform"

# -- 2. gradient spot-check over every parameter ------------------------------
loss.backward()
h, tol, n_probe = 1e-5, 1e-6, 5
worst, ok = 0.0, True
params = model.parameters()
for pi, p in enumerate(params):
    flat = p.data.reshape(-1)
    gflat = p.grad.reshape(-1)
    for j in rng.choice(flat.size, size=min(n_probe, flat.size), replace=False):
        old = flat[j]
        flat[j] = old + h
        lp = float(model(idx, tgt)[1].data)
        flat[j] = old - h
        lm = float(model(idx, tgt)[1].data)
        flat[j] = old
        num = (lp - lm) / (2 * h)
        ana = gflat[j]
        # rel tolerance + abs floor: FD cancellation noise on a ~3.14 loss is
        # ~1e-10 absolute, which dominates rel err when the true grad is ~1e-5
        rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-8)
        worst = max(worst, rel)
        if rel > tol and abs(ana - num) > 1e-9:
            ok = False
            print(f"[FAIL] param{pi} entry {j}: analytic {ana:.3e} numeric {num:.3e}")

print(f"spot-checked {len(params)} params x {n_probe} entries, "
      f"worst rel err {worst:.2e}")
print("MODEL GRAD CHECK PASSED" if ok else sys.exit("MODEL GRAD CHECK FAILED"))
