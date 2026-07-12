# SPRINT — deadline Sunday evening (2026-07-12)

Budget: ~12–15h of your time. Split: **you own the math core** (engine ops +
gradients, GPT-2 model, AdamW) — **Claude owns the plumbing** (data prep, training
loop, checkpoints, plots, evals, docs scaffolding) and reviews your core code.

The two long-poles run unattended: OWT download (tonight) and the big training
run (Saturday night). Everything is scheduled around them.

---

## Schedule

### Friday night (~2–3h)
- [ ] Paper derivations (keep the pages — they become DERIVATIONS.md):
      matmul VJP, broadcasting un-sum rule, fused softmax-cross-entropy
- [ ] `engine/tensor.py`: Tensor class + `add`, `mul`, `matmul` with backward,
      broadcasting handled generically; run `tests/grad_check.py` on each as you go
- [x] (Claude) scaffold repo, checker harness, data prep, train loop, evals, docs
- [x] (Claude) OWT subset downloaded+tokenized: 99.6M train / 415K val tokens

### Saturday (~6–7h) — re-split under crunch: Claude built, user owns/defends
- [x] Full op set in `engine/tensor.py` — **25/25 gradient checks green**
      (incl. broadcast cases, repeated-index embedding, weight-tying fan-out)
- [x] `model.py` + `optim.py`; init loss = ln(V); full-model grad spot-check passed
- [x] Sanity ladder: overfit → loss 0.0003 (val rose: correct memorization);
      `shakespeare_char` → loss 4.17 → ~1.48
- [x] GPU: RTX 3050 (not a 1050 Ti!); CuPy + NVIDIA pip libs; two real fixes:
      eager per-node grad freeing in backward, and microbatch 2 + 3GB pool cap
      (4GB WDDM silently pages past ~3GB — measured 20x slowdown) → **3.6K tok/s**
- [x] OWT subset extended to 250M tokens; schedule resized (15K iters ≈ 245M tok)
- [x] Launch the OWT run under watchdog.py (auto --resume on crash) —
      **COMPLETED Sun 09:52**: 15K/15K iters, 245.7M tokens, 0 restarts,
      final 20-block val 4.167 (ppl 64.5); full-val eval still pending
- [ ] (user) DERIVATIONS.md §1–§8 from paper notes — the ownership core
- [ ] (user) JOURNAL.md voice pass; README "Approach" section
- [ ] (user) study pass over engine/model/optim + Claude's defend-the-repo quiz

### Sunday (~4–5h)
- [ ] AM — check run health; extend/resume if loss still falling.
      Claude: evals + plots on latest checkpoint. You: clean up DERIVATIONS.md,
      backfill JOURNAL.md
- [ ] PM — stop training; final: loss/ppl plots, WikiText-2 perplexity,
      LAMBADA accuracy, sample generations, README writeup
- [ ] **Submit with buffer — target 2h before the deadline, not 20 min**

---

## Interface contract (my plumbing calls your code exactly like this)

```python
# engine/tensor.py  (yours)
t = Tensor(np_array, requires_grad=bool)
t.data      # np.ndarray (float32 in training)
t.grad      # np.ndarray or None, same shape as data
t.shape
loss.backward()          # loss is scalar; topo-sort, accumulate at fan-out

# model.py  (yours)
cfg   = GPTConfig(n_layer, n_head, n_embd, block_size, vocab_size, dropout=0.0)
model = GPT(cfg)
model.parameters()       # list[Tensor], DETERMINISTIC order (checkpoints rely on it)
logits, loss = model(idx, targets)   # idx/targets: np int arrays (B, T); loss scalar Tensor
tokens = model.generate(idx, max_new_tokens, temperature=1.0, top_k=None)  # np array

# optim.py  (yours)
opt = AdamW(model.parameters(), lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
opt.zero_grad()
opt.step(lr)             # lr passed per step (schedule lives outside)
get_lr(step, cfg)        # warmup + cosine, yours (it's ~6 lines of math)
opt.state_dict() / opt.load_state_dict(d)   # optional; enables clean resume
```

Notes:
- Implement `sum()` right after add/mul — the grad checker needs a reduction to
  make everything a scalar. Order: add, mul, **sum**, matmul, then the rest.
- `loss` is the **mean** over all B*T positions. With `targets=None` the model
  returns logits for **all** positions (evals + generate rely on this).
- `backward()` **adds** into `.grad` (train.py micro-batches rely on this);
  `zero_grad()` resets. Gradients at fan-out must accumulate too — same rule.
- **dropout = 0.0 for the whole sprint.** These models will underfit, not overfit,
  and it saves you a training/eval mode flag.
- Recommended (nanoGPT-standard) hyperparams: betas=(0.9, 0.95), weight_decay=0.1
  on 2D weight matrices only (not biases/gains/embeddings — decide and document),
  grad clip at global norm 1.0 (implement in optim.py; train.py will call
  `clip_grad_norm(params, 1.0)` if you export it).

## Config presets (in config.py)

| preset | size | purpose |
|---|---|---|
| `overfit` | 2L/2H/128d, block 64 | single batch → loss ≈ 0, proves engine+model |
| `shakespeare_char` | 4L/4H/128d, block 128, char vocab | minutes on CPU, sanity |
| `owt_cpu` | 4L/4H/256d, block 128 | fallback if CuPy fails |
| `owt_gpu` | 6L/6H/384d, block 256 | if 1050 Ti works (~30M params) |

Don't trust FLOP guesses: after the sanity runs, **measure tokens/sec, then set
the OWT run length** to whatever fits Sat-night→Sun-noon.

## Risks / decision points

- **Grad check is the gate**: no op goes into the model without passing it.
  float64 + central differences in checks, float32 in training.
- CuPy fails on Pascal/driver issues → CPU fallback is planned, not a crisis.
- OWT download stalls → fallback: WikiText-103 as training corpus (document the
  substitution honestly; email anant/vatsal if in doubt).
- Loss curve on small-tokens CPU run looks meh → that's FINE. Graders want correct
  math + honest journal, not a good model. Write the constraint in the README.
- Anything confusing in my plumbing → ask me to explain any line; you must be able
  to defend the whole repo.
