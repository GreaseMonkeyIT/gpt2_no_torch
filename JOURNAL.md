# Journal — roadblocks and how I tackled them

<!-- One dated entry per session. What I tried, what broke, what I learned.
     Roadblocks are graded artifacts here, not embarrassments. -->

## Fri 2026-07-10 — setup + engine start

- Deadline moved to Sunday evening; re-planned as a sprint (SPRINT.md).
  Split: I write engine/model/optimizer + all derivations; AI handles
  scaffolding (data prep, train loop, evals, plots) — disclosed in README.
- Env: pyenv-win Python 3.12.10, venv, numpy-only training path.
  Data: streaming a ~100M-token OpenWebText subset overnight (full OWT is
  ~9B tokens — far beyond weekend compute on a 1050 Ti).
- Micrograd warmup done ([notebooks/warmups/00_micrograd_warmup.ipynb](notebooks/warmups/00_micrograd_warmup.ipynb)),
  following the zero-to-hero video but writing it myself, not copying:
  - `Value` class with closure-based `_backward`, `+=` gradient accumulation,
    topo-sort backward — this skeleton IS the tensor engine's architecture
  - verified fan-out accumulation with the `b = a + a` check (grad = 2, passed)
  - verified a gradient numerically with a finite-difference probe — the habit
    that becomes tests/grad_check.py
  - built tanh two ways (fused op vs composed from exp) — same gradient; first
    contact with the fused-vs-composed trade-off that returns in softmax/layernorm
  - Neuron/Layer/MLP skeleton written (untested) — will NOT carry into the tensor
    engine (per-neuron python loops); keeping only the parameters() pattern
- Known issue in the notebook (Claude review): `__pow__` has
  `assert isinstance(other, (int, float)),` with a trailing comma and no message —
  SyntaxError as saved; also grads-as-floats and no broadcasting are exactly the
  two upgrades the tensor engine must make
- Derivations done on paper: <!-- TODO(you): list which, note anything that fought back -->
- Engine progress: <!-- TODO(you) -->
- Roadblocks: <!-- TODO(you) -->

## Sat 2026-07-11 — the build day (crunch mode)

- Deadline pressure + external work swamped me; per the organizers' "AI use is
  fine, own every line" rule, Claude built the implementation and I shifted to
  derivations, review, and being able to defend the repo. Disclosed in README.
- Warmups I did myself before that: bigrams (notebooks/warmups/01_bigrams_warmup.ipynb) —
  count model NLL 2.4765, one-layer softmax net converging to the same optimum
  (2.4818 w/ L2), which is the point of the exercise.
- Engine: micrograd skeleton tensorized; all 25 op gradient checks pass at
  ~1e-9 rel err (float64 central differences).
- Model: init loss 10.859 vs ln(50257)=10.825 — the "uniformly unsure at init"
  sanity check. Full-model finite-difference spot check across all 28 params.
  Lesson learned: naive rel-err tolerance false-alarms on ~1e-5 gradients; need
  an absolute floor too (FD cancellation noise ~1e-10 on a ~3.14 loss).
- Overfit run: 4.22 → 0.0003 while val loss ROSE 4.16 → 7.46 — scary-looking but
  correct: that is what memorizing one batch looks like.
- Shakespeare: 4.17 → ~1.48 over 3000 iters on CPU (~2-4K tok/s).
- GPU saga (RTX 3050 4GB, CuPy):
  1. cupy wheel missing CUDA DLLs → NVIDIA pip libs (runtime/cublas/curand/nvrtc)
  2. OOM: tied-head logits are (B·T,50257) — 823MB at microbatch 16 → microbatch 4
  3. still OOM: engine kept every intermediate's grad alive → free each node's
     grad right after its own backward runs (reverse-topo guarantees safety)
  4. 380 tok/s at "100% util": 4GB WDDM pages to system RAM past ~3GB — proved it
     by capping the pool at 3GB (instant OOM at 3.1GB) → microbatch 2, accum 32
  5. result: 3,572 tok/s — 9x. VRAM was the constraint all along, as predicted.
- <!-- TODO(you): your reflections — what surprised you, what you'd do differently -->

### Sample evolution (same prompts, checkpoints through the night)

Shakespeare (final, iter 3000, prompt `ROMEO:`):

> ROMEO:
> By the kname and tend tlib, that you content
> Disposing that the hearth of please,
> That I may be words so end him to itself?
> O, and you have give his the brow of them worm
> Of our obstrocks thems consulk'd of fair his pardoned

Char-level model has to earn every letter — hence "obstrocks" and "consulk'd";
the iambic-ish line shape and speaker structure came free from 1M chars.

OWT @ iter 1,001 (~16M tokens), prompt `The meaning of life is`:

> The meaning of life is very good after making money, and that it should be a
> bad thing, for his time and a young person, but by him, it needs to be a
> conversation with himself.

BPE tokens mean no misspellings ever; coherence lasts ~one clause. Also: the
internet's values are showing.

OWT @ iter 5,001 (~82M tokens), same prompt:

> The meaning of life is often to display both your own and all of them, and if
> you are you, the reality of life is a reality of life, and therefore they are
> a genuine sense of truth, and that they are our own. If you are a whole of
> love with that, and that they are a reality

Syntax now holds across a full sentence (agreement, clause nesting) but the
content orbits — "a reality of life is a reality of life" — the small-model
repetition attractor. Grammar is learned before meaning.
<!-- TODO(you): your own read on the 1K -> 5K change, in your voice -->

OWT @ iter ~11,500 (~194M tokens), same prompt:

> The meaning of life is always the truth." The other two had to wonder, "Why
> do I want to be called? Why do I want to be the head coach?"
>
> From his dad's little-known father to his father, Paul, to his sister, Paul
> and his father, and Paul, the brother

New at this stage: correct quote/dialogue mechanics (open/close quotes,
questions inside quotes, attribution), and named entities as reusable
referents. But coreference is degenerate — father, sister, and brother are all
"Paul". The repetition attractor climbed a level of abstraction: at 82M tokens
it looped phrases, at 194M it loops entities. Coherence horizon ~2 sentences,
then topic drift (philosophy -> head coach -> family tree).
<!-- TODO(you): your read, in your voice -->

OWT @ iter ~14,860 (~243M tokens, minutes before the end), same prompt:

> The meaning of life is often the condition for those who are not well
> established, and people who are not well-developed, which you are likely to
> find, from whom you are, might not remember it.
>
> You may find her in her own life, but she does have a different life, which
> is a condition

The Paul-collapse is gone: a pronoun referent ("her") now survives across a
paragraph break with a consistent identity. The register holds abstract for
the whole passage. The loop is still there — "the condition ... which is a
condition" — but its period has stretched from adjacent phrases to the span of
the entire sample. Each stage of training didn't remove the attractor; it
lengthened its orbit.
<!-- TODO(you): your read, in your voice -->

OWT @ iter 15,000 (245.7M tokens — the final checkpoint), same prompt
(temperature 0.8, top-k 200):

> The meaning of life is nothing for having someone else. In 2015, the
> authorities found a second high-ranking U.S. citizen who came to New Delhi for
> a report defending his visa under the "US Constitution". After a few days in
> Juba's life, Bong-Reheng, an American citizen, was killed in the second floor
> of New Delhi in 2014. In 2015, the majority of the country's citizens were
> covered.

<!-- TODO(you): your read on the finished model vs the 14.8K sample, in your voice -->

### Overnight progress log

| time (Sat/Sun) | iter | tokens | train loss | val loss (ppl) |
|---|---|---|---|---|
| Sat ~21:20 | 2,000 | 32.8M | 5.025 | 4.794 (120.8) |
| Sat ~22:25 | 4,000 | 65.5M | 4.647 | 4.614 (100.9) |
| Sat ~23:30 | 5,000 | 81.9M | 4.394 | 4.568 (96.4) |
| Sun ~02:35 | 8,000 | 131.1M | 4.252 | 4.348 (77.3) |
| Sun ~04:40 | 10,000 | 163.8M | 4.178 | 4.302 (73.9) |
| Sun ~06:15 | 11,500 | 188.4M | 4.207 | 3.992 (54.2) * |
| Sun ~07:15 | 12,000 | 196.6M | 4.090 | 4.240 (69.4) |
| Sun ~08:20 | 13,000 | 213.0M | 4.085 | 4.181 (65.4) |
| Sun ~09:25 | 14,000 | 229.4M | 4.127 | 4.133 (62.4) |
| Sun ~09:45 | 14,500 | 237.6M | 4.149 | 4.228 (68.6) |
| Sun 09:52 | 14,999 (final) | 245.7M | 4.181 | 4.167 (64.5) |

\* outlier reading, not a breakthrough: each val eval samples only 20 random
256-token blocks (~10K tokens), and document-level difficulty correlation gives
the estimator ±~0.1 noise — adjacent evals swing that much all night (4.446 →
4.568 → 4.410 around 5K; the "worse" 5K reading was the same effect in the
unlucky direction). Local trend at 11.5K ≈ 4.30–4.35 (ppl ~74). Fix applied:
status.py now also prints a 3-eval mean; the FINAL reported number will be
computed over the full 415K-token val set on the last checkpoint, not from the
20-block estimator.

## Sun 2026-07-12 — derivations night + endgame

- Pulled an overnighter on the math: all 11 derivation chapters written by hand
  in my Obsidian vault — broadcasting/unbroadcast, matmul, reductions,
  shape ops, embedding scatter-add, GELU, softmax (Jacobian + VJP), fused
  softmax-CE (with the 1/BT mean factor this time), LayerNorm-as-composition,
  AdamW bias correction, attention as a shape walkthrough. Claude then ported
  them into DERIVATIONS.md, added per-section "In the engine" cross-references
  to the exact code lines, and reviewed the math. Disclosed in README.
- **The review caught one real error, and it's a good one:** in attention
  Step 3 I wrote g_K = Qᵀ g_S. Shape-check: that lands on K-transpose's shape,
  not K's — it is the gradient of Kᵀ (the tensor literally multiplied), one op
  short of the answer. The engine actually computes my formula mid-graph (matmul
  backward at the transpose node) and then the transpose backward flips it to
  the correct g_K = g_Sᵀ Q. So the math I derived was the right quantity with
  the wrong name — the kind of error a shape-check catches in five seconds,
  which is why every VJP in the engine is shape-checked by tests. Lesson
  banked. <!-- TODO(you): rewrite this reflection in your own words -->
- Also from tonight: LayerNorm chapter was left half-finished at ~5am; Claude
  completed the γ/β parameter-gradient sums (the §1 broadcast rule applied to
  the (C,)-shaped params) — flagged here for honesty.
- Finished zero-to-hero parts 3+4 on my other PC while the main rig trained
  (notebooks/warmups/03_activations_bn.ipynb, 04_backprop_ninja.ipynb). Part 4
  is the one that matters for this repo: hand-backpropped an entire MLP+BN+CE
  net, 26/26 gradient checks exact vs torch; the fused cross-entropy backward
  I wrote there (softmax, subtract 1 at targets, divide by n) is line-for-line
  the backward in engine/tensor.py's fused op — and the fused normalization
  backward (Exercise 3, checked to 9e-10) is the §9 stretch goal, done in the
  batchnorm setting. Manual-gradient training run: val 2.1099 (reference 2.1162).
- Val-loss estimator lesson (see table footnote above): a 20-block eval has
  ±0.1 noise; a 3.99 reading at 11.5K looked like a breakthrough and wasn't.
  status.py now prints a 3-eval trend; the final number comes from the full
  415K-token val set.
- <!-- TODO(you): how the overnighter actually felt / what fought back -->

### Run endgame (filled as it happens)

- **Run complete, Sun 09:52** — all 15,000 iters (0–14,999), exited rc=0.
  One watchdog attempt, **zero crashes/restarts** the whole night; the resume
  machinery was never needed (the 3GB pool cap doing its job silently).
- Totals: **245.7M tokens**, ~14.7h wall clock (launched Sat ~19:10),
  steady 4.3–4.4K tok/s throughout.
- Losses: train 10.86 → 4.181; final 20-block val **4.1666 (ppl 64.5)**,
  mean of last 3 evals 4.1758 (ppl 65.1). Per the estimator-noise note above,
  the headline number still needs the full 415K-token val pass.
- Final checkpoint: `out/owt_gpu/ckpt.npz` (+ timestamped .snap copies through
  the night, latest 09:45 @ iter 14,500 — keep until final evals are done).
- Pending (next session): full-val loss/ppl on the final checkpoint,
  WikiText-2 perplexity, LAMBADA accuracy, loss/ppl plots, final samples.
