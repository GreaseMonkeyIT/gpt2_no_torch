# GPT-2 from scratch — no torch/jax autograd

Rumik Polaris fellowship assignment. Train a GPT-2 style decoder-only transformer
on OpenWebText with **hand-derived gradients** (no autograd from torch/jax).
Reference scaffold: [nanoGPT](https://github.com/karpathy/nanoGPT).

**Rule I'm holding myself to: I own every line. This file is structure only — all
derivations and code are mine.**

---

## Decision 0 — pick the approach (do this first, on paper)

- [ ] **Option A: per-module manual backward.** Each layer (Linear, LayerNorm,
      Attention, …) is a class with `forward()` and `backward(grad_out)`. You chain
      them by hand. Simple, explicit, CS231n-style.
- [ ] **Option B: minimal tensor autograd.** A `Tensor` class wrapping `np.ndarray`
      that records ops into a graph and runs `backward()` via topological sort
      (micrograd, but for tensors). More upfront work, but you still derive every
      op's gradient — and it covers both things the assignment lists.

**Recommendation: Option B.** You derive the same math either way, but B forces you
to solve broadcasting/reduction gradients once, generically — and it reads as the
stronger submission. If B stalls, A is a safe fallback with reusable derivations.

- [ ] Stack decision: numpy (BLAS-backed matmul is the workhorse). Note GPU option:
      CuPy is a near-drop-in numpy replacement if CPU training is too slow — decide
      later, keep the code array-library-agnostic where easy.
- [ ] Tokenizer: `tiktoken` GPT-2 BPE (allowed — it's not autograd), same as nanoGPT.

---

## Phase 1 — math groundwork (paper + notebook, no repo code yet)

Watch/skim from the referenced Karpathy playlist: micrograd, and *"becoming a
backprop ninja"* (makemore part 4) — the latter is literally this assignment in
miniature.

Derive on paper, from scratch (keep the paper — it becomes the derivations doc):

- [ ] Matmul: `dL/dA` and `dL/dB` for `C = A @ B`, including the batched case
- [ ] Broadcasting: what happens to gradients when shapes were broadcast in forward
      (the un-broadcast / sum-over-broadcast-axes rule)
- [ ] Softmax (row-wise), and why you fuse it with cross-entropy
- [ ] Cross-entropy loss w.r.t. logits (the clean fused form)
- [ ] LayerNorm w.r.t. input, gain, bias — the trickiest one; do it slowly
- [ ] GELU (decide: exact `erf` form vs tanh approximation — derive the one you use)
- [ ] Embedding lookup (gradient is a scatter-add — why?)
- [ ] Attention as a composition: `softmax(QKᵀ/√d + mask) @ V` — you don't need one
      monolithic formula if your primitives compose, but understand the flow

**Exit criteria:** every formula above written in my own hand, with shapes annotated.

---## Phase 2 — the engine + gradient checker (build the checker FIRST)

The gradient checker is what makes "doing it myself" safe: it catches every wrong
derivation mechanically.

- [ ] `check_grad(f, x)`: central-difference numerical gradient vs my analytical one
      (use float64 for checks; know the tolerance you accept and why)
- [ ] Tensor/op core, only the ops GPT-2 actually needs:
  - [ ] add, mul, matmul (with broadcasting)
  - [ ] reshape / transpose / split / concat / slice (for the QKV head plumbing)
  - [ ] sum, mean, max (for numerically stable softmax)
  - [ ] exp, log, tanh (or erf), power/sqrt
  - [ ] embedding lookup (gather → scatter-add backward)
  - [ ] fused softmax-cross-entropy
  - [ ] (decide) LayerNorm as fused op vs composed from primitives — fused is faster
        and a good derivation showcase
- [ ] Topological-sort backward pass, gradient accumulation at fan-out nodes
- [ ] Unit test per op: analytical vs numerical grad, on random shapes incl. broadcasts
- [ ] Optional extra check: compare against PyTorch gradients **in tests only**
      (allowed — torch never touches the training path)

**Exit criteria:** every op passes gradient check, including weird broadcast shapes.

## Phase 3 — GPT-2 model

Mirror nanoGPT's `model.py` structure so reviewers can map it:

- [ ] Config dataclass (n_layer, n_head, n_embd, block_size, vocab_size, dropout)
- [ ] Token + positional embeddings
- [ ] CausalSelfAttention (multi-head, causal mask, single QKV projection)
- [ ] MLP (4x expansion, GELU)
- [ ] Block (pre-norm residual: `x + attn(ln(x))`, `x + mlp(ln(x))`)
- [ ] LM head with **weight tying** to token embedding (mind the gradient: two paths
      into one parameter)
- [ ] GPT-2 init scheme (normal 0.02, scaled residual projections)
- [ ] `generate()` for sampling (forward-only — cheap and great for the report)
- [ ] Shape-annotate every tensor in comments — the assignment explicitly values this

**Exit criteria:** forward pass produces sane loss at init (~ln(vocab_size) ≈ 10.82
for GPT-2's 50257 vocab — know why), full-model gradient check passes on a tiny config.

## Phase 4 — optimizer + training loop

All hand-rolled:

- [ ] AdamW (bias correction, decoupled weight decay — decide which params get decay)
- [ ] LR schedule: linear warmup + cosine decay
- [ ] Gradient clipping (global norm)
- [ ] Gradient accumulation (to fake bigger batches)
- [ ] Training loop: eval every N steps, checkpoint save/resume (npz), loss logging to
      a plain file/CSV so plots are reproducible

## Phase 5 — sanity ladder (do NOT touch OpenWebText yet)

- [ ] Overfit a single batch to ~zero loss — the classic "everything works" test
- [ ] Train char-level tiny Shakespeare (nanoGPT's small config) — loss curve should
      roughly track what nanoGPT reports
- [ ] (optional but powerful) Same tiny config in PyTorch with identical init and data
      order; loss curves should overlap for the first ~100 steps

**Exit criteria:** Shakespeare samples look like Shakespeare-ish text.

## Phase 6 — OpenWebText + scale (the grind — journal everything)

- [ ] nanoGPT's `data/openwebtext/prepare.py` approach: tokenize once to memmapped
      `train.bin` / `val.bin`; random-offset batch sampling
- [ ] Pick a model size that's honest about hardware: numpy on CPU ≈ tens of
      GFLOPs — think ~10–30M params on a subset, not 124M on the full set.
      State the compute constraint openly in the report; they're grading math, not GPUs.
- [ ] Profile: matmul should dominate; kill python-loop hotspots, preallocate,
      float32 everywhere in training
- [ ] GPU plan for the 1050 Ti (Pascal sm_61, 4 GB VRAM, FP32-only — no FP16, no
      Triton). Staged, each stage only after gradient checks pass:
  1. numpy CPU reference stays the source of truth for gradient checks (float64)
  2. CuPy drop-in via the `xp = numpy|cupy` pattern → matmuls on cuBLAS
  3. fused CUDA C kernels via `cupy.RawKernel` for the memory-bound ops
     (softmax, LayerNorm, cross-entropy) — profile first, journal the speedups
  4. (optional) study llm.c for kernel structure; full C/CUDA port only if time
- [ ] Mind 4 GB VRAM: small microbatch + gradient accumulation; activations dominate
- [ ] Train the final run: log train/val loss every eval interval, checkpoint regularly

## Phase 7 — evals + artifacts (what they explicitly asked for)

- [ ] Train/val **loss curves** + **perplexity** (= exp(loss), state that)
- [ ] Downstream benchmark, pick one that measures language modeling:
      **LAMBADA** (last-word accuracy — the GPT-2 paper's headline LM eval) or
      **WikiText-2/103 perplexity** or **HellaSwag** (nanoGPT uses it). Compare
      against published GPT-2 numbers with an honest note about the size gap.
- [ ] Sample generations at several checkpoints (shows learning progression nicely)
- [ ] Gradient-check results table (op → max error) — proof the math is right

## Phase 8 — writeup + submission

- [ ] README: approach chosen and why, how to run, results
- [ ] DERIVATIONS.md (or PDF of handwritten pages): the Phase 1 math, cleaned up
- [ ] JOURNAL.md → "roadblocks and how I tackled them" section (they asked for this)
- [ ] Eval tables + plots embedded
- [ ] Email anant@rumik.ai / vatsal@rumik.ai with any clarifying questions **early**
      (e.g., is a scaled-down model on an OpenWebText subset acceptable — almost
      certainly yes, but asking shows judgment)

---

## JOURNAL.md — start it on day 1

One dated entry per session: what I tried, what broke, what I learned. This is a
graded artifact, not a diary. Roadblocks are points here, not embarrassments.

## Known traps (check when stuck)

- Broadcasting backward: forgetting to sum gradients over broadcast dimensions —
  the #1 source of silent wrong gradients
- Softmax without max-subtraction → NaN at scale
- LayerNorm backward sign/term errors — trust the gradient checker, not your eyes
- Causal mask: use a large negative *finite* value, or handle `-inf` carefully in backward
- Weight tying: gradient must accumulate from both the embedding and the LM head
- Adam without bias correction → mysterious early-training behavior
- float64 for gradient *checks*, float32 for *training* — mixing these up hides bugs
- Fan-out nodes (residual streams!) must **accumulate** gradients, not overwrite

## Suggested repo layout

```
rumik/
  engine/        # Tensor + ops + backward (the heart)
  model.py       # GPT-2 on top of engine
  optim.py       # AdamW, schedule, clipping
  train.py       # loop, logging, checkpoints
  data/          # prepare scripts, bins (gitignored)
  tests/         # gradient checks per op + full model
  evals/         # lambada / wikitext / hellaswag harness
  plots/
  DERIVATIONS.md
  JOURNAL.md
  README.md
```
