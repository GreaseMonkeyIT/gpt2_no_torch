"""GPT-2 on the numpy autograd engine. Structure mirrors nanoGPT's model.py.

No gradient code here — every backward comes from engine op VJPs, composed
automatically by the graph (that's the whole point of the autograd approach).
Weight tying note: wte is used twice (token embedding AND transposed LM head);
the engine's fan-out accumulation sums both gradient paths — verified by the
"weight tying fan-out" check in tests/test_ops.py.
"""

import math
from dataclasses import dataclass

import numpy as np

from engine import tensor as et
from engine.tensor import (Tensor, embedding, gelu, softmax,
                           softmax_cross_entropy, to_numpy)


@dataclass
class GPTConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout: float = 0.0  # accepted for config compat; unused (sprint trim)


def _param(rng, *shape, std=0.02):
    return Tensor((rng.standard_normal(shape) * std).astype(np.float32),
                  requires_grad=True)


class Linear:
    def __init__(self, n_in, n_out, rng, std=0.02):
        self.w = _param(rng, n_in, n_out, std=std)
        self.b = Tensor(np.zeros(n_out, dtype=np.float32), requires_grad=True)

    def __call__(self, x):                      # (..., n_in) -> (..., n_out)
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]


class LayerNorm:
    """Composed from engine primitives; backward chains automatically (§9)."""

    def __init__(self, n):
        self.g = Tensor(np.ones(n, dtype=np.float32), requires_grad=True)
        self.b = Tensor(np.zeros(n, dtype=np.float32), requires_grad=True)

    def __call__(self, x):                      # (B, T, C)
        mu = x.mean(axis=-1, keepdims=True)     # (B, T, 1)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        xhat = xc * ((var + 1e-5) ** -0.5)
        return xhat * self.g + self.b

    def parameters(self):
        return [self.g, self.b]


class CausalSelfAttention:
    def __init__(self, cfg, rng):
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = Linear(cfg.n_embd, 3 * cfg.n_embd, rng)
        # GPT-2 init: residual projections scaled down by 1/sqrt(2·n_layer)
        self.c_proj = Linear(cfg.n_embd, cfg.n_embd, rng,
                             std=0.02 / math.sqrt(2 * cfg.n_layer))
        # additive causal mask: 0 on/below diagonal, -1e9 above. A large FINITE
        # negative, not -inf: inf produces nan in the softmax backward (0·inf)
        self.mask = et.xp.asarray(
            np.triu(np.full((cfg.block_size, cfg.block_size), -1e9,
                            dtype=np.float32), k=1)[None, None])

    def __call__(self, x):
        B, T, C = x.shape
        H, hs = self.n_head, C // self.n_head
        qkv = self.c_attn(x)                                  # (B, T, 3C)
        q, k, v = (qkv[:, :, i * C:(i + 1) * C] for i in range(3))
        q = q.reshape(B, T, H, hs).transpose(1, 2)            # (B, H, T, hs)
        k = k.reshape(B, T, H, hs).transpose(1, 2)
        v = v.reshape(B, T, H, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B, H, T, T)
        att = att + self.mask[:, :, :T, :T]                   # constant, no grad
        att = softmax(att, axis=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)        # merge heads
        return self.c_proj(y)

    def parameters(self):
        return self.c_attn.parameters() + self.c_proj.parameters()


class MLP:
    def __init__(self, cfg, rng):
        self.c_fc = Linear(cfg.n_embd, 4 * cfg.n_embd, rng)
        self.c_proj = Linear(4 * cfg.n_embd, cfg.n_embd, rng,
                             std=0.02 / math.sqrt(2 * cfg.n_layer))

    def __call__(self, x):
        return self.c_proj(gelu(self.c_fc(x)))

    def parameters(self):
        return self.c_fc.parameters() + self.c_proj.parameters()


class Block:
    """Pre-norm residual block: x + attn(ln(x)), x + mlp(ln(x))."""

    def __init__(self, cfg, rng):
        self.ln_1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, rng)
        self.ln_2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg, rng)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def parameters(self):
        return (self.ln_1.parameters() + self.attn.parameters()
                + self.ln_2.parameters() + self.mlp.parameters())


class GPT:
    def __init__(self, cfg, seed=42):
        rng = np.random.default_rng(seed)
        self.cfg = cfg
        self.wte = _param(rng, cfg.vocab_size, cfg.n_embd)   # tied LM head too
        self.wpe = _param(rng, cfg.block_size, cfg.n_embd)
        self.blocks = [Block(cfg, rng) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(cfg.n_embd)

    def parameters(self):
        # deterministic order — checkpoint.py and AdamW state rely on it
        ps = [self.wte, self.wpe]
        for blk in self.blocks:
            ps += blk.parameters()
        return ps + self.ln_f.parameters()

    def __call__(self, idx, targets=None):
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= self.cfg.block_size
        tok = embedding(self.wte, idx)                        # (B, T, C)
        pos = embedding(self.wpe, np.arange(T)[None])         # (1, T, C), broadcasts
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = x @ self.wte.transpose(0, 1)                 # (B, T, V), tied
        loss = softmax_cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, seed=1337):
        """Forward-only sampling; plain numpy on logits, no graph needed."""
        rng = np.random.default_rng(seed)
        idx = np.asarray(idx)
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            lg = to_numpy(logits.data)[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, lg.shape[-1])   # top_k may exceed a small vocab
                kth = np.partition(lg, -k, axis=-1)[:, [-k]]
                lg = np.where(lg < kth, -np.inf, lg)
            lg = lg - lg.max(axis=-1, keepdims=True)
            p = np.exp(lg)
            p /= p.sum(axis=-1, keepdims=True)
            nxt = np.array([[rng.choice(p.shape[-1], p=row)] for row in p])
            idx = np.concatenate([idx, nxt], axis=1)
        return idx
