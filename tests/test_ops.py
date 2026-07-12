"""Op-by-op gradient battery: every DERIVATIONS.md section, mechanically verified.

Run: .venv\\Scripts\\python.exe tests\\test_ops.py   (from repo root)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.tensor import embedding, gelu, softmax, softmax_cross_entropy  # noqa: E402
from tests.grad_check import check  # noqa: E402

rng = np.random.default_rng(7)
r = []

# §0 elementwise + §1 broadcasting
r.append(check(lambda a, b: (a + b).sum(), [(4, 5), (5,)], name="add bcast (5,)"))
r.append(check(lambda a, b: (a + b).sum(), [(2, 3, 4), (3, 1)], name="add bcast (3,1)"))
r.append(check(lambda a, b: (a * b).sum(), [(2, 3, 4), (4,)], name="mul bcast"))
r.append(check(lambda a, b: (a - b).sum(), [(3, 4), (3, 1)], name="sub bcast"))
r.append(check(lambda a: ((a * 3.0 + 1.5) / 2.0).sum(), [(3, 4)], name="const arith"))
r.append(check(lambda a: ((a * a + 1.2) ** -0.5).sum(), [(3, 4)], name="pow -0.5 (rsqrt)"))
r.append(check(lambda a: a.exp().sum(), [(3, 4)], name="exp"))
r.append(check(lambda a: (a * a + 0.5).log().sum(), [(3, 4)], name="log"))
r.append(check(lambda a: a.tanh().sum(), [(3, 4)], name="tanh"))

# §2 matmul
r.append(check(lambda a, b: (a @ b).sum(), [(4, 5), (5, 3)], name="matmul 2d"))
r.append(check(lambda a, b: (a @ b).sum(), [(2, 3, 4), (4, 5)], name="matmul bcast weight"))
r.append(check(lambda a, b: (a @ b).sum(), [(2, 2, 3, 4), (2, 2, 4, 3)], name="matmul batched"))

# §3 reductions
r.append(check(lambda a: (a.sum(axis=1, keepdims=True) * a).sum(), [(3, 4)], name="sum keepdims"))
r.append(check(lambda a: (a.mean(axis=-1, keepdims=True) * a).sum(), [(2, 3, 4)], name="mean keepdims"))
r.append(check(lambda a: a.sum(axis=0).tanh().sum(), [(3, 4)], name="sum axis0 chained"))

# §4 data movement
r.append(check(lambda a: a.reshape(2, 3, 2, 2).transpose(1, 2).tanh().sum(),
               [(2, 6, 2)], name="reshape+transpose"))
r.append(check(lambda a: (a[:, 1:3] * a[:, 1:3]).sum(), [(3, 5)], name="slice"))

# §5 embedding (repeated ids must accumulate)
idx = np.array([[0, 1, 1, 2], [2, 2, 0, 4]])
r.append(check(lambda w: (embedding(w, idx) * embedding(w, idx)).sum(),
               [(5, 3)], name="embedding gather (repeats)"))

# §6 gelu
r.append(check(lambda a: gelu(a).sum(), [(3, 4)], name="gelu"))

# §7 softmax (weighted so the gradient is non-degenerate)
r.append(check(lambda a, w: (softmax(a, axis=-1) * w).sum(), [(3, 5), (3, 5)],
               name="softmax"))
# shift invariance: same weighted output when inputs are shifted by a constant
r.append(check(lambda a, w: (softmax(a + 3.7, axis=-1) * w).sum(), [(3, 5), (3, 5)],
               name="softmax shifted"))

# §8 fused softmax-CE (int targets captured in closure)
tgt = rng.integers(0, 8, size=(2, 3))
r.append(check(lambda lg: softmax_cross_entropy(lg, tgt), [(2, 3, 8)],
               name="fused softmax-CE"))

# §9 layernorm composed from primitives (weighted against constants)
C = rng.standard_normal((2, 3, 4))


def layernorm_expr(x, g, b):
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    xhat = xc * ((var + 1e-5) ** -0.5)
    return ((xhat * g + b) * C).sum()


r.append(check(layernorm_expr, [(2, 3, 4), (4,), (4,)], name="layernorm composed"))

# weight tying: same tensor as embedding AND (transposed) output head — fan-out
tie_idx = np.array([[0, 2], [1, 1]])
r.append(check(lambda w: (embedding(w, tie_idx) @ w.transpose(0, 1)).tanh().sum(),
               [(4, 3)], name="weight tying fan-out"))

# residual stream: same tensor through two paths (the transformer's skip pattern)
r.append(check(lambda x, w: (x + gelu(x @ w)).sum(), [(3, 4), (4, 4)],
               name="residual fan-out"))

print()
if all(r):
    print(f"ALL {len(r)} CHECKS PASSED")
else:
    sys.exit(f"{r.count(False)}/{len(r)} FAILED — fix before building on this")
