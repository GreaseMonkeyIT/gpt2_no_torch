"""AdamW, warmup+cosine schedule, global-norm clipping — all hand-rolled.

Bias correction (DERIVATIONS §10): EMAs initialized at zero are biased toward
zero; E[m_t] = (1-beta^t)·E[g], so dividing by (1-beta^t) undoes it exactly.
The W in AdamW: weight decay applied directly to weights (decoupled), not mixed
into the gradient — and only to matrices/embeddings, not biases/LN gains.
"""

import math

import numpy as np

from engine import tensor as et


def get_lr(step, cfg):
    if step < cfg["warmup_iters"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_iters"]
    if step >= cfg["lr_decay_iters"]:
        return cfg["min_lr"]
    r = (step - cfg["warmup_iters"]) / (cfg["lr_decay_iters"] - cfg["warmup_iters"])
    return cfg["min_lr"] + 0.5 * (1 + math.cos(math.pi * r)) * (cfg["lr"] - cfg["min_lr"])


def clip_grad_norm(params, max_norm):
    total = math.sqrt(sum(float((p.grad ** 2).sum())
                          for p in params if p.grad is not None))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return total


class AdamW:
    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [et.xp.zeros_like(p.data) for p in self.params]
        self.v = [et.xp.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.b1
            m += (1.0 - self.b1) * g          # first-moment EMA
            v *= self.b2
            v += (1.0 - self.b2) * g * g      # second-moment EMA
            if self.wd and p.data.ndim >= 2:  # decay matrices/embeddings only
                p.data *= 1.0 - lr * self.wd
            p.data -= lr * (m / bc1) / (et.xp.sqrt(v / bc2) + self.eps)

    def state_dict(self):
        d = {f"m{i}": m for i, m in enumerate(self.m)}
        d.update({f"v{i}": v for i, v in enumerate(self.v)})
        d["t"] = np.array(self.t)
        return d

    def load_state_dict(self, d):
        self.t = int(d["t"])
        self.m = [et.xp.asarray(d[f"m{i}"]) for i in range(len(self.params))]
        self.v = [et.xp.asarray(d[f"v{i}"]) for i in range(len(self.params))]
