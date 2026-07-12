# Code map — every file and function, and why it exists

Study companion for defending the repo. Three layers: **the math core**
(engine — where all graded content lives), **the model** (GPT-2 assembled from
engine ops), and **plumbing** (training, data, evals — proves nothing about
calculus, just makes the core run).

The one-paragraph mental model: `Tensor` wraps a numpy array. Every op returns
a new `Tensor` carrying a closure (`_backward`) that knows how to push
gradients to its inputs — a VJP. `backward()` sorts the graph topologically
and runs the closures in reverse. Everything else in the repo is either
"call ops in the right order" (model) or "feed batches and update weights"
(training).

---

## Layer 1 — the math core

### engine/tensor.py — the autograd engine (~330 lines, THE file)

| name | what it does |
|---|---|
| `xp` | module-global array backend: numpy by default, cupy after `use_gpu()`. Ops look it up at call time, so one codebase runs CPU and GPU. |
| `use_gpu()` | swaps `xp` to cupy. Must run before any Tensor exists. |
| `to_numpy(a)` | device→host escape hatch (`.get()` on cupy arrays, no-op on numpy). Used by checkpointing and sampling. |
| `_scatter_add(target, idx, src)` | indexed `+=`: `np.add.at` on CPU, `cupyx.scatter_add` on GPU (cupy has no `add.at`). |
| `unbroadcast(g, shape)` | THE broadcasting rule (§1): sums `g` over every axis numpy stretched, because broadcasting is implicit copying and a copy is a fan-out. Called by every elementwise backward and matmul. |
| `_accum(t, g)` | adds a VJP contribution into `t.grad`, creating it lazily. `+=` is why fan-out (residuals, weight tying) just works. |
| `Tensor.__init__` | wraps data; `.grad` starts `None`; `_prev` (inputs) + `_backward` (closure) form the graph. |
| `__add__`, `__mul__` | elementwise ops. Two paths: Tensor⊕Tensor (both get gradients) vs Tensor⊕constant (int/float/ndarray — no gradient; used for the causal mask and scalar scales). Both un-broadcast. |
| `__neg__`, `__sub__`, `__truediv__` | composed from add/mul — zero new derivations (micrograd trick). Division by constants only. |
| `__pow__(n)` | constant exponents only; powers `n·x^(n-1)`. Gives us sqrt/rsqrt for LayerNorm. |
| `__matmul__` | §2: `dA = g @ Bᵀ`, `dB = Aᵀ @ g` (last two axes swapped), then un-broadcast — which is what sums a weight's gradient over the batch. |
| `sum(axis, keepdims)` | §3: backward re-expands the collapsed axis and broadcasts the gradient back over it. |
| `mean(...)` | implemented as `sum × (1/n)` — composition instead of a new derivation. |
| `reshape`, `transpose` | §4: data moves, values don't; backward is the exact inverse move. |
| `__getitem__` | slicing (used for QKV split); backward scatter-adds the gradient into a zeros array of the input shape. |
| `exp`, `log`, `tanh` | §0 scalar ops, vectorized; each reuses its own output in backward (cache trick from the micrograd notebook). |
| `backward()` | iterative topo-sort (explicit stack, no recursion limit), seeds `grad=1`, runs closures in reverse; **frees each intermediate's `.grad` right after its own closure runs** — reverse order guarantees no one needs it anymore. That line halved GPU memory. |
| `gelu(x)` | fused §6: GPT-2's tanh-approx GELU; fused (not composed) to avoid ~8 extra (B,T,4C) intermediates. |
| `softmax(x, axis)` | fused §7: max-subtraction is legal by shift invariance; backward is `p·(g − Σg·p)`. Used by attention. |
| `softmax_cross_entropy(logits, targets)` | fused §8: mean NLL via log-softmax (log-sum-exp stable). Backward is the famously simple `(p − onehot)/N`, done by indexed subtraction — no onehot matrix ever built. |
| `embedding(w, idx)` | §5: row gather; backward **scatter-adds** because a token id appearing k times is a k-way fan-out (assignment would silently drop k−1 gradients). |

### engine/__init__.py
Re-exports the public names so `from engine import Tensor` works.

---

## Layer 2 — the model

### model.py — GPT-2, mirrors nanoGPT's structure

| name | what it does |
|---|---|
| `GPTConfig` | dataclass of hyperparams (`dropout` accepted but unused — sprint trim, models this size underfit anyway). |
| `_param(rng, *shape, std)` | normal(0, std) float32 parameter, wrapped in a Tensor. |
| `Linear` | `x @ W + b`. The `+ b` broadcasts — its gradient works because of `unbroadcast`. |
| `LayerNorm` | **composed** from mean/sub/mul/pow primitives (§9) — no hand-derived fused backward; the engine chains it. Knows the subtlety: μ and σ are functions of x (three gradient paths, summed automatically). |
| `CausalSelfAttention` | one QKV projection, split, reshape to (B,H,T,hs), `QKᵀ/√hs`, **additive mask of −1e9** (finite! −inf makes 0·inf = nan in softmax backward), softmax, `@V`, merge heads, output projection. `c_proj` init scaled by 1/√(2·n_layer) — GPT-2's variance control for deep residual stacks. |
| `MLP` | Linear → gelu → Linear, 4× expansion. Same scaled init on the projection. |
| `Block` | pre-norm residuals: `x + attn(ln(x))`, `x + mlp(ln(x))`. The `x +` is a fan-out; `_accum` handles it. |
| `GPT.parameters()` | flat list in **deterministic order** — checkpoints and Adam state are stored positionally, so this order is a contract. |
| `GPT.__call__(idx, targets)` | token emb + position emb (broadcast over batch), blocks, final LN, then `logits = x @ wteᵀ` — **weight tying**: wte is used twice, and fan-out accumulation sums both gradient paths. Returns (logits, loss) with logits for all positions. |
| `GPT.generate(...)` | forward-only sampling loop in plain numpy (temperature, top-k clamped to vocab); no KV cache — O(T²) per token, fine for demos. |

---

## Layer 3 — plumbing

### optim.py — the update rule
| name | what it does |
|---|---|
| `get_lr(step, cfg)` | linear warmup → cosine decay to `min_lr`. |
| `clip_grad_norm(params, max)` | global L2 norm across all grads; rescales if over. |
| `AdamW` | per-param EMAs `m` (gradient) and `v` (gradient²); update `m̂/(√v̂+ε)` with **bias correction** `1/(1−βᵗ)` (§10 — EMAs start at 0 and are biased toward 0 early). Weight decay is **decoupled** (multiplies weights directly, the "W") and applied only to ndim≥2 params (matrices/embeddings, not biases/LN gains). `state_dict/load_state_dict` for resume. |

### train.py — the loop
| name | what it does |
|---|---|
| `get_batch(data, T, B, rng)` | random offsets into the memmapped token stream; y is x shifted by one. No epochs — random sampling. |
| `main()` | picks preset; `use_gpu()` if the preset says so; builds model/optimizer; optional `--resume` from checkpoint; then: for each iter — `get_lr`, accumulate `grad_accum` microbatches (backward ADDS into grads), divide grads by accum count, clip, `opt.step(lr)`; every `eval_interval` measure val loss + save checkpoint; log CSV rows; Ctrl+C saves before exit. |

### checkpoint.py
`save` = params (+ Adam state) as npz with config JSON embedded, converted to numpy (`to_numpy`) so GPU checkpoints load anywhere. `load` = the reverse, positional against `parameters()` order.

### config.py
The four presets: `overfit` (prove the pipeline), `shakespeare_char` (CPU sanity), `owt_cpu` (fallback), `owt_gpu` (the real run: microbatch 2 × accum 32 because 4GB WDDM pages past ~3GB — measured 20× slowdown; run with `CUPY_GPU_MEMORY_LIMIT=3GB` to fail loudly instead).

### watchdog.py / status.py / sample.py
`watchdog.py`: relaunches training with `--resume` on crash (max 5 restarts).
`status.py`: progress bar, ppl, ETA from the CSV.
`sample.py`: load checkpoint → `generate()` → decode (char meta.pkl or tiktoken).

### data/prepare_shakespeare.py, data/prepare_openwebtext.py
Shakespeare: download → char vocab → uint16 bins + meta.pkl.
OWT: HF streaming → tiktoken GPT-2 BPE → uint16 bins, `<|endoftext|>` between
docs, every 200th doc to val. Streamed subset because full OWT is ~9B tokens.

---

## The referee — tests

| file | what it proves |
|---|---|
| `tests/grad_check.py` | `numeric_grad`: central differences (O(h²) truncation), float64, h=1e-5. `check`: builds Tensors, runs YOUR backward, compares per input. Self-test runs without the engine. |
| `tests/test_ops.py` | 25 checks = every DERIVATIONS section as an executable claim, incl. broadcast shapes, repeated embedding ids, weight-tying fan-out, residual fan-out. |
| `tests/test_model.py` | init loss ≈ ln(V) (uniformly-unsure check) + finite-difference spot-probe of every parameter through the whole transformer. Rel tolerance + absolute floor (FD cancellation noise dominates near-zero grads). |

## Evals & plots

`evals/eval_lm.py`: `wikitext2_ppl` (non-overlapping T-windows, honest cheap
version) and `lambada_acc` (greedy exact-match of the final word's BPE tokens).
`plots/make_plots.py`: loss + perplexity PNGs from a run's CSV (`ema` smoothing,
palette/chrome constants, direct end-label on val).

---

## One training step, traced

1. `get_batch` → `x, y` int arrays (B,T) from the memmap
2. `model(x, y)`: gather embeddings → 6 × Block (each: LN → attention → residual add → LN → MLP → residual add) → final LN → tied-head matmul → fused softmax-CE → scalar loss Tensor. Every op appended its closure to the graph.
3. `loss.backward()`: topo-sort ~1500 nodes, run closures newest→oldest; grads land in `p.grad` for all 76 parameters; intermediate grads freed as it goes
4. Repeat 2–3 for each microbatch — `_accum`'s `+=` merges the microbatches
5. `p.grad /= grad_accum` → `clip_grad_norm` → `opt.step(lr)` (Adam math, decay, update)
6. Every 500 iters: val loss over 20 batches, checkpoint, CSV row

## DERIVATIONS.md ↔ code cross-reference

| § | claim | lives in |
|---|---|---|
| 0 | scalar ops | `exp/log/tanh/__pow__` |
| 1 | un-broadcast rule | `unbroadcast()` |
| 2 | matmul VJP | `__matmul__` |
| 3 | reductions | `sum`/`mean` |
| 4 | data movement | `reshape/transpose/__getitem__` |
| 5 | gather → scatter-add | `embedding()` |
| 6 | GELU derivative | `gelu()` |
| 7 | softmax + shift invariance | `softmax()` |
| 8 | fused softmax-CE | `softmax_cross_entropy()` |
| 9 | LayerNorm composed | `model.LayerNorm` |
| 10 | Adam bias correction | `AdamW.step` (`bc1`, `bc2`) |
| 11 | attention composition | `CausalSelfAttention.__call__` |

## Interviewer bait — know these cold

- Why must gradients **sum** at fan-out? (chain rule over multiple paths; `b=a+a` from your own notebook)
- Why sum over broadcast axes? (broadcast = copy = fan-out)
- Why is the mask −1e9 and not −inf? (inf·0 → nan in softmax's VJP)
- Why scatter-**add** in embedding backward? (repeated token ids)
- Why does weight tying need no special code? (`wte` appears twice in the graph; `_accum` sums both paths)
- Why divide Adam's EMAs by (1−βᵗ)? (zero-init bias; expand the recursion)
- Why float64 in checks but float32 in training? (FD needs precision; training needs speed/memory)
- Why free intermediate grads during backward? (reverse topo order ⇒ consumers already ran; halves peak memory)
- Why microbatch 2 on a 4GB card? (tied-head logits are (B·T)×50257 floats; WDDM pages silently past ~3GB)
