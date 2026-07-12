# GPT-2 from scratch — no torch/jax autograd

Rumik Polaris fellowship assignment: a GPT-2 style decoder-only transformer,
trained on an OpenWebText subset, with a hand-written numpy autograd engine.
All gradients derived by hand — see [DERIVATIONS.md](DERIVATIONS.md).
Roadblocks and process: [JOURNAL.md](JOURNAL.md).

<!-- TODO(you): 2-3 sentences in your own voice: approach chosen and why -->

## Approach

<!-- TODO(you): engine design (tensor autograd, topo-sort backward), what ops
     exist, what is fused and why, compute constraints (1050 Ti / CPU) -->

### Design choices — what and why

| choice | instead of | why |
|---|---|---|
| LayerNorm | BatchNorm (my warmups) | normalizes per position over channels — no cross-example coupling (batch statistics are garbage at microbatch 2), no running stats / train-eval mode; biased 1/C variance per GPT-2. Derivation §9; the fused BN backward hand-verified in warmups/04. |
| GELU (tanh approx) | tanh / ReLU | GPT-2's activation; smooth with nonzero gradient for small negatives. Derivative derived in §6, fused in the engine for memory. |
| learned positional embeddings (wpe) | nothing (MLP warmup needed none) | attention is permutation-invariant — the fixed-window MLP got position free from its concat slots; a transformer must inject it explicitly. |
| causal self-attention | fixed-window concat (makemore MLP) | every position attends over all previous ones instead of a hard 3-token window; mask is −1e9, finite, because −inf breeds NaN in the softmax backward (§11). |
| pre-norm residual blocks | plain stacked layers | the identity path keeps gradients alive through depth; a residual add is a fan-out, handled by the engine's `+=` accumulation rule (§1). |
| weight tying (wte = LM head) | separate output matrix | halves embedding parameters; the two gradient paths sum automatically at fan-out — no special code. |
| AdamW + warmup/cosine + clipping | SGD + step decay (my warmups) | per-parameter step scaling across embeddings vs matrices; bias correction derived in §10; decoupled decay on 2D params only (decaying LN gains fights the normalization); warmup + clipping for stability at lr 6e-4. |
| init std 0.02, residual projections ×1/√(2·n_layer) | gain-based Kaiming (my warmups) | controls variance growth along the residual stream with depth, GPT-2's scheme, rather than per-layer variance preservation. |
| fused softmax-CE, GELU, softmax, embedding | composing from primitives | the analytic cancellation (−y/p against p) done on paper once, exactly — the composed backward is numerically noisy and memory-hungry at V=50257 (§8). |

## Results

**Model:** 30.0M parameters (6 layers, 6 heads, 384 embd, block 256), trained from
scratch on a 245M-token OpenWebText subset. That is ~1/4 the parameters and ~1/40
the training tokens of GPT-2 124M (WebText, ~10B tokens). It underfits by design
(see SPRINT.md / JOURNAL.md) — the graded artifact is correct hand-derived
gradients, not a competitive model.

Final training loss 4.18; **full validation perplexity 63.0** (nll 4.1433 over the
entire 1.12M-token OWT val set — the noisy 20-block training estimator read 64.5).

![owt_gpu train/val loss](plots/owt_gpu_loss.png)
![owt_gpu train/val perplexity](plots/owt_gpu_ppl.png)

Downstream zero-shot evals (forward-only; honest simplifications documented in
`evals/eval_lm.py` and `evals/full_val.py`):

| eval | this model (30M, 245M tok) | GPT-2 124M (paper) |
|---|---|---|
| OWT val ppl (full set) | 63.0 | — |
| WikiText-2 ppl | 203.9 | 29.41 |
| LAMBADA acc (500 ex) | 7.8% | 45.99% |

The two downstream numbers sit far from GPT-2 124M — expected at 1/4 the params,
1/40 the tokens, and (for WikiText-2) a corpus out of the training distribution.
Random-baseline LAMBADA is ~0%, so 7.8% is genuine last-word prediction, just weak.

### Sample generations

`python sample.py out/owt_gpu/ckpt.npz --prompt "..."` (temperature 0.8, top-k 200):

> **The meaning of life is** nothing for having someone else. In 2015, the
> authorities found a second high-ranking U.S. citizen who came to New Delhi for a
> report defending his visa under the "US Constitution". After a few days in Juba's
> life, Bong-Reheng, an American citizen, was killed in the second floor of New
> Delhi in 2014. In 2015, the majority of the country's citizens were covered.

> **The history of science shows that** science is quite distinct from the last 10
> years that a lot of science is now in a climate, so we are seeing more important
> real science possibilities from the very science we represent [...]

Coherent within a sentence, drifting in topic after one or two — the small-model
behavior traced checkpoint-by-checkpoint in JOURNAL.md.

## How to run

```
pip install -r requirements.txt          # numpy training path; no torch/jax
python data/prepare_shakespeare.py       # sanity dataset (seconds)
python data/prepare_openwebtext.py --max-tokens 250000000   # 248.9M train / 1.12M val
python tests/test_ops.py                 # gradient checks, every op
python train.py shakespeare_char        # sanity run
python train.py owt_gpu                  # main run (owt_cpu for CPU fallback)
python sample.py out/owt_gpu/ckpt.npz --prompt "The meaning of life is"
python evals/eval_lm.py out/owt_gpu/ckpt.npz --task wikitext2
```

## AI use disclosure

Engine, model, optimizer, and all gradient derivations: written by me.
Scaffolding (data prep, training-loop plumbing, plots, eval harness) built with
Claude's help; I own and can defend every line.
