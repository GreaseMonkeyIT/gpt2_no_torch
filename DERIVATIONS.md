# Gradient derivations

Scalar warm-ups were derived in [notebooks/warmups/00_micrograd_warmup.ipynb](notebooks/warmups/00_micrograd_warmup.ipynb)
(micrograd-style, worked through in the notebook while following the zero-to-hero guide). 
Tensor VJPs below were derived by hand in my Obsidian notes, then ported and
consolidated here (with Opus). Everything is verified 
mechanically by `tests/test_ops.py`, which was in part inspired by [RyanTomich/np_GPT2](https://github.com/RyanTomich/np_GPT2)
(central finite differences, float64). AI review pass disclosed in the README:
one real error was caught by Claude in #11 (the `g_K` formula; see the note there), #9 was
completed by Claude, and the "In the engine" cross-references were added.

## Notation

- `L`: the final scalar loss. For any node `x`, its gradient is `g_x = ∂L/∂x`,
  always the same shape as `x`.
- Every `backward` answers one question: **given `g_out`, produce `g_in`**; a
  vector–Jacobian product. No full Jacobian is ever materialized.
- Fan-out rule: a node used in k places receives the **sum** of k contributions
  (`+=` in code). Verified in the warmup with `b = a + a → a.grad = 2`.
- Shapes: `B` batch, `T` block/sequence, `C` n_embd, `V` vocab, `H` heads,
  `hs = C/H` head size.

## 0. Scalar ops (derived in the warmup notebook)

| op | forward | local derivative | backward line |
|---|---|---|---|
| add | `out = a + b` | `1` w.r.t. each | `g_a += g_out; g_b += g_out` |
| mul | `out = a * b` | the *other* operand | `g_a += b * g_out; g_b += a * g_out` |
| pow (const n) | `out = a**n` | `n·a^(n−1)` | `g_a += n * a**(n-1) * g_out` |
| exp | `out = e^a` | `out` itself | `g_a += out * g_out` |
| tanh | `out = tanh(a)` | `1 − out²` | `g_a += (1 - out**2) * g_out` |
| neg / sub / div | composed: `−a = a·(−1)`, `a−b = a+(−b)`, `a/b = a·b^(−1)` | — | no new derivation |

Note carried to tensors: exp and tanh backwards reuse `out` — cache the forward
result, never recompute.

## 1. Broadcasting — the un-broadcast rule

Modern tensor libraries allow tensors of different shapes to participate in the
same elementwise operation through **broadcasting**. Broadcasting behaves as if
the smaller tensor were copied to match the larger tensor, although no physical
copy is made. During backpropagation, this implicit copying must be reversed.

### Broadcasting is fan-out

Suppose $x = 5$ is broadcast to length three. Mathematically this behaves as
though $[x, x, x]$ were created — the computational graph is a three-way
fan-out, each child carrying the same value.

Warm-up (already proved in #0's notebook): for $b = x + x$, the add node has
**two edges** from $x$, and the chain rule sums over edges:

$$
\frac{\partial L}{\partial x}
= \frac{\partial L}{\partial b}\frac{\partial b}{\partial x}
= 2\,\frac{\partial L}{\partial b}.
$$

General fan-out: if broadcasting creates $k$ copies $x \to (x_1,\ldots,x_k)$,
each copy contributes independently to the loss. By the multivariable chain
rule, with $\partial x_i/\partial x = 1$ for every copy,

$$
\boxed{\;\frac{\partial L}{\partial x} = \sum_{i=1}^{k}\frac{\partial L}{\partial x_i}.\;}
$$

This is the mathematical reason gradients are **summed** during backpropagation.

### The three broadcast cases

**Vector to batch.** $x \in \mathbb{R}^{C}$ broadcast to $y \in \mathbb{R}^{B\times T\times C}$
(a bias added to activations): $y_{btc} = x_c$, so each $x_c$ appears $B \times T$
times, and

$$
\boxed{\;g_x[c] = \sum_{b=1}^{B}\sum_{t=1}^{T} g_y[b,t,c].\;}
$$

The channel dimension is preserved; the broadcast dimensions disappear through
summation.

**Singleton dimensions.** $(1, C) \to (B, C)$: the size-1 axis is reused per
batch element, so $g_x[0,c] = \sum_b g_y[b,c]$ — sum over the stretched axis,
**keeping** it as size 1.

**Added leading dimensions.** $(C) \to (B, T, C)$: the new axes did not exist in
$x$, so they are summed **away** entirely.

### The unbroadcast theorem

Every broadcasted element is an independent copy of the original value, and
broadcasting only creates copies along (1) newly added leading axes and
(2) axes whose original size is 1. Reversing broadcasting therefore consists
precisely of summing the gradient over those axes:

$$
\boxed{\;g_x = \mathrm{unbroadcast}(g_y,\ \mathrm{shape}(x)).\;}
$$

Specification of `unbroadcast(g, s)`: sum over every leading axis added during
broadcasting; sum (with `keepdims`) over every axis where $s$ has size 1;
result has shape $s$. The spec is independent of which elementwise op produced
the broadcast.

**In the engine:** [tensor.py:48](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L48) implements the spec
verbatim (the `while` loop is case 3, the `keepdims=True` sum is case 2), and
[`_accum`](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L61) is the fan-out sum itself — every VJP
contribution is `+=`'d, never assigned. Every binary op's backward calls both:
see `__add__` ([tensor.py:87](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L87)) and `__mul__`
([tensor.py:104](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L104)).

## 2. Matmul

Let $C = AB$ with $A \in \mathbb{R}^{m\times k}$, $B \in \mathbb{R}^{k\times n}$,
forward definition $C_{ij} = \sum_{p} A_{ip}B_{pj}$, upstream $g_C = \partial L/\partial C$.

### Gradient w.r.t. A

By the multivariable chain rule, only row $i$ of $C$ depends on $A_{ip}$:

$$
\frac{\partial L}{\partial A_{ip}}
= \sum_{j=1}^{n} \frac{\partial L}{\partial C_{ij}}\frac{\partial C_{ij}}{\partial A_{ip}}
= \sum_{j=1}^{n} g_C(i,j)\, B_{pj},
$$

which is exactly the $(i,p)$ entry of a matrix product:

$$
\boxed{\;g_A = g_C B^{\mathsf T}.\;}
$$

### Gradient w.r.t. B

Symmetrically, only column $j$ depends on $B_{pj}$, and
$\partial C_{ij}/\partial B_{pj} = A_{ip}$:

$$
\frac{\partial L}{\partial B_{pj}} = \sum_{i=1}^{m} A_{ip}\, g_C(i,j)
\qquad\Longrightarrow\qquad
\boxed{\;g_B = A^{\mathsf T} g_C.\;}
$$

Sanity check that never lies: $g_A$ must have $A$'s shape $(m,k)$ — and
$(m,n)\cdot(n,k)$ is the only way to build it from $g_C$ and $B$.

### Batched matmul (our Linear layers)

For $A \in \mathbb{R}^{B\times m\times k}$ against a shared weight
$W \in \mathbb{R}^{k\times n}$, the weight is mathematically **broadcast**
across the batch dimension. By the unbroadcast theorem (#1), each batch
contributes an independent gradient:

$$
\boxed{\;g_W = \sum_{b=1}^{B} A_b^{\mathsf T}\, g_{C,b}.\;}
$$

**In the engine:** [tensor.py:144](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L144). The transposes are
`swapaxes(-1, -2)` (last two axes — the batched generalization of $^{\mathsf T}$),
and the $\sum_b$ above is not special-cased: it falls out of the same
`unbroadcast` call every other op uses. #1 pays for itself here.

## 3. Reductions: sum, mean, max

### Sum

$y = \sum_i x_i$: since $\partial y/\partial x_i = 1$,

$$
\boxed{\;\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y}\;}
$$

— every input receives the same upstream gradient. For an axis reduction
$Y_i = \sum_j X_{ij}$, likewise $g_X[i,j] = g_Y[i]$: the gradient is
**broadcast back** over the collapsed axis.

### Mean

$y = \frac{1}{n}\sum_i x_i$ gives $\partial y/\partial x_i = \frac1n$:

$$
\boxed{\;g_x = \frac{1}{n}\mathrm{broadcast}(g_y)\;}
$$

— identical to sum, scaled by $1/n$.

### Max

$y = \max(x)$ routes gradient only to the selected element:
$g_{x_i} = g_y$ if $x_i = \max(x)$, else $0$ (ties: any subgradient choice is
valid).

**In the engine:** sum at [tensor.py:159](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L159) (re-expand the
collapsed axis, then `broadcast_to`); mean at
[tensor.py:171](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L171) is literally `sum × (1/n)` — no separate
derivation, matching this section. **Max never got an op**: the only max in the
model is the stability shift inside softmax/cross-entropy, which is applied to
raw `.data` as a *constant* — legalized by shift invariance (#7). The scaffold's
"max only if needed" resolved to *not needed*.

## 4. Reshape / transpose / split / slice

These operations do not modify numerical values; they only rearrange them.
Their backward pass is therefore the exact inverse rearrangement.

- **Reshape:** every output element corresponds to exactly one input element,
  so $\boxed{g_x = \mathrm{reshape}(g_y, \mathrm{shape}(x))}$.
- **Transpose:** $B = A^{\mathsf T}$, $B_{ij} = A_{ji}$
  $\Rightarrow \boxed{g_A = g_B^{\mathsf T}}$ — swap the indices back.
- **Slice:** only the sliced region influences the loss: zeros of the input
  shape, with $g_y$ pasted into the sliced region.
- **Split:** forward partitions; backward concatenates the pieces' gradients.

**In the engine:** reshape [tensor.py:177](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L177), transpose
[tensor.py:186](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L186) (`swapaxes` is its own inverse),
slice/gather via `__getitem__` [tensor.py:195](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L195) — whose
backward is a scatter-**add** into zeros, which #5 explains.

## 5. Embedding gather

Embedding layers **select** rows from $W \in \mathbb{R}^{V\times C}$ rather than
multiplying by it. Forward, with integer indices $\mathrm{idx} \in \{0..V{-}1\}^{B\times T}$:

$$
\mathrm{out}_{btc} = W_{\mathrm{idx}_{bt},\,c}.
$$

By the multivariable chain rule, only terms with $\mathrm{idx}_{bt} = v$ and $k = c$
survive:

$$
\boxed{\;\frac{\partial L}{\partial W_{vc}} = \sum_{\,b,t\;:\;\mathrm{idx}_{bt} = v} g_{\mathrm{out}}(b,t,c).\;}
$$

### Why scatter-add, never scatter-assign

If the same token appears twice, `idx = [3, 3]`, then $W_3$ fans out to two
outputs, and the chain rule requires $g_{W_3} = g_1 + g_2$. Indexed
**assignment** overwrites one contribution — NumPy's fancy-index write keeps
only the last value, silently dropping gradients for every repeated token
(and in a language-model batch, common tokens repeat constantly).

**In the engine:** [`_scatter_add`](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L40) — `np.add.at` on CPU,
`cupyx.scatter_add` on GPU — used by both the fused
[`embedding`](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L319) op and `__getitem__`'s backward. The
repeated-index case has its own test in the grad-check battery.

## 6. GELU (tanh approximation)

GPT-2's activation:

$$
\mathrm{GELU}(x) = \tfrac12 x\left(1 + \tanh(u)\right),
\qquad
u(x) = a\,(x + 0.044715\,x^3),
\qquad
a = \sqrt{2/\pi}.
$$

Product rule on $f(x) = \tfrac12 x$ times $g(x) = 1 + \tanh(u)$, then chain
rule through $u$ with $u'(x) = a\,(1 + 0.134145\,x^2)$ (note $0.134145 = 3 \times 0.044715$)
and $\frac{d}{dx}\tanh(u) = (1 - \tanh^2 u)\,u'$:

$$
\boxed{\;
\frac{d}{dx}\mathrm{GELU}(x)
= \underbrace{\tfrac12\left(1+\tanh u\right)}_{\text{product rule: } f'g}
+ \underbrace{\tfrac12 x\,(1-\tanh^2 u)\;a\,(1 + 0.134145\,x^2)}_{\text{chain rule: } f g'}
\;}
$$

**In the engine:** [tensor.py:263](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L263), fused so the graph
holds one node instead of eight. The forward's $\tanh(u)$ is cached (`t`) and
reused in backward — the #0 rule about never recomputing.

## 7. Softmax (standalone — attention needs it)

$$
p_i = \frac{e^{z_i}}{\sum_k e^{z_k}}.
$$

### Shift invariance

For any constant $c$:

$$
\mathrm{softmax}(z - c)_i
= \frac{e^{z_i - c}}{\sum_k e^{z_k - c}}
= \frac{e^{z_i}\,e^{-c}}{e^{-c}\sum_k e^{z_k}}
= \mathrm{softmax}(z)_i.
$$

Subtracting the row max therefore changes nothing mathematically while keeping
every exponent $\le 0$ (float32 `exp` overflows past $z \approx 88.7$). It also
means the subtracted max may be treated as a **constant** in backward — the
gradient through the shift cancels exactly, so no max-op gradient (#3) is ever
needed.

### Jacobian

Self term (quotient rule): $\partial p_i/\partial z_i = p_i(1 - p_i)$.
Cross term ($i \ne j$): $\partial p_i/\partial z_j = -p_i p_j$.
Combined with the Kronecker delta:

$$
\boxed{\;\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j).\;}
$$

### Vector–Jacobian product

Contract the Jacobian with the upstream $g_p$ — two terms, one diagonal, one
from the shared normalizer:

$$
g_z[j] = \sum_i g_p[i]\, p_i(\delta_{ij} - p_j)
= p_j\, g_p[j] - p_j \sum_i g_p[i]\, p_i,
$$

$$
\boxed{\;g_z = p \odot \left(g_p - \langle g_p, p\rangle\,\mathbf 1\right).\;}
$$

**In the engine:** [tensor.py:279](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L279) — forward does the max
shift on `.data` (constant, per the invariance argument), backward is the boxed
line verbatim: `p * (g - (g * p).sum(axis, keepdims=True))`.

## 8. Fused softmax + cross-entropy — the star

Softmax $p_i = e^{z_i}/\sum_k e^{z_k}$; cross-entropy for one-hot targets
$L = -\sum_i y_i \log p_i$.

### Derivation

Chain rule through the probabilities, using #7's Jacobian and
$\partial L/\partial p_i = -y_i/p_i$:

$$
\frac{\partial L}{\partial z_j}
= \sum_i \left(-\frac{y_i}{p_i}\right) p_i(\delta_{ij} - p_j)
= -\sum_i y_i(\delta_{ij} - p_j)
= -\sum_i y_i \delta_{ij} + p_j \sum_i y_i.
$$

For one-hot labels $\sum_i y_i = 1$ and $\sum_i y_i \delta_{ij} = y_j$, hence

$$
\boxed{\;\frac{\partial L}{\partial z_j} = p_j - y_j.\;}
$$

If the loss is **averaged** over $B \times T$ positions (ours is),

$$
\boxed{\;\frac{\partial L}{\partial z} = \frac{p - y}{B\,T}.\;}
$$

### Why this must be fused

Notice what cancelled: the $-y_i/p_i$ factor. At initialization
$p \approx 1/V = 1/50257$, so the *composed* backward (softmax op, then log op,
then mean) would carry intermediate gradients of magnitude $\sim 50{,}000$
before cancelling them against $p_i$ — analytically exact, numerically noisy,
and it materializes a $(B{\cdot}T, V)$ probability tensor plus its gradient.
Doing the cancellation on paper, once, exactly — that is this section — is why
the fused op exists.

**In the engine:** [tensor.py:294](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/engine/tensor.py#L294).
Three bridges from the math to the code:

- *Integer targets, no one-hot:* $y$ is never materialized. Since $p - y$
  differs from $p$ only at the target index, the backward is
  `d[arange(n), y] -= 1.0` — that line **is** $p - y$ for an implicit one-hot.
- *The mean:* `d /= n` with $n = B{\cdot}T$ — the second box.
- *Stability:* forward computes **log-softmax** via the log-sum-exp identity
  $\log p_i = (z_i - m) - \log\sum_k e^{z_k - m}$ with $m = \max_k z_k$
  (shift legal by #7), so no probability is ever exponentiated and then logged.
  `exp(logp)` in backward recovers $p$ exactly.

Independent verification: I re-derived and hand-wrote this same backward
(softmax, subtract 1 at the targets, divide by n) against PyTorch's autograd in
[notebooks/warmups/04_backprop_ninja.ipynb](notebooks/warmups/04_backprop_ninja.ipynb)
— matches `F.cross_entropy`'s gradient to ~5e-9.

## 9. LayerNorm — composed, not fused (sprint trim)

LayerNorm is deliberately **composed from primitives** the engine already has —
mean, subtract, multiply, `**(-0.5)` — so its backward emerges from #0–#3
automatically and needs no fused derivation. Forward, per position over the
channel axis:

$$
\mu = \frac1C\sum_i x_i,
\qquad
\sigma^2 = \frac1C\sum_i (x_i - \mu)^2,
\qquad
\hat x_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}},
\qquad
y_i = \gamma_i \hat x_i + \beta_i.
$$

The subtlety worth stating: $\mu$ and $\sigma$ are **functions of $x$**, so the
automatic backward is silently applying the total derivative — gradient flows
to $x$ along three paths (directly, through $\mu$, and through $\sigma^2$), and
the topo-sorted engine sums them (#1 fan-out rule) without being told.

Parameter gradients, as a sanity check on what the engine computes: per
position $\partial L/\partial \beta_i = g_{y_i}$ and
$\partial L/\partial \gamma_i = g_{y_i}\hat x_i$; since $\gamma, \beta \in \mathbb{R}^{C}$
are broadcast over $(B, T)$, the #1 rule sums those contributions:

$$
\boxed{\;g_\beta[c] = \sum_{b,t} g_y[b,t,c],
\qquad
g_\gamma[c] = \sum_{b,t} g_y[b,t,c]\;\hat x[b,t,c].\;}
$$

**In the engine/model:** [model.py:54](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/model.py#L54) — five lines of primitives.
Note the variance is the biased $1/C$ mean (matching GPT-2), $\varepsilon = 10^{-5}$
sits inside the square root, and the $(\cdot)^{-0.5}$ is the #0 pow rule doing
the square root and the division in one op. Stretch goal (fused three-term
backward + benchmark) was correctly triaged away — the composed version is what
shipped and what the grad checks verify. The fused three-term normalization
backward *was* however derived and verified by hand in the batchnorm setting
(same total-derivative structure: direct path, mean path, variance path) in
[notebooks/warmups/04_backprop_ninja.ipynb](notebooks/warmups/04_backprop_ninja.ipynb),
Exercise 3 — exact to ~1e-9 against torch.

## 10. AdamW bias correction

Adam maintains EMAs of the gradient's first and second moments. Both start at
zero, so early estimates are biased toward zero; bias correction removes this
exactly.

First moment: $m_t = \beta m_{t-1} + (1-\beta) g_t$, $m_0 = 0$. Unrolling,

$$
m_t = (1-\beta)\left(g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots + \beta^{t-1} g_1\right).
$$

For stationary gradients ($E[g_k] = E[g]$), the geometric series
$\sum_{k=0}^{t-1}\beta^k = \frac{1-\beta^t}{1-\beta}$ gives

$$
\boxed{\;E[m_t] = (1-\beta^t)\,E[g]
\qquad\Longrightarrow\qquad
\hat m_t = \frac{m_t}{1-\beta^t}.\;}
$$

The second moment is the same argument with $\beta_2$ and $g_t^2$:
$\hat v_t = v_t/(1-\beta_2^t)$.

Update with **decoupled** weight decay (the W in AdamW):

$$
\boxed{\;\theta \leftarrow \theta - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} - \eta\lambda\theta.\;}
$$

Decay acts on the weights directly, not through the gradient — so it is not
rescaled by $1/\sqrt{\hat v}$, which is the entire difference from Adam-with-L2.

**In the engine:** [optim.py:51](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/optim.py#L51). The decay is applied as
`p.data *= 1 - lr*wd` (identical to the $-\eta\lambda\theta$ term), and **only
to tensors with `ndim >= 2`** — matrices and embeddings decay; biases and
LayerNorm gains don't (decaying a LN gain toward zero fights the normalization
it parameterizes). `lr` arrives per-step from the warmup+cosine schedule in
[`get_lr`](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/optim.py#L16).

## 11. Attention as a composition

Multi-head self-attention is not a primitive. It is a sequence of operations
whose backwards were all derived above — the engine chains them; this section
walks the shapes and names which VJP fires at each arrow.

Input $X \in \mathbb{R}^{B\times T\times C}$, $C = H\cdot hs$.

1. **QKV projection** — $Q = XW_Q$, $K = XW_K$, $V = XW_V$ (one fused Linear in
   practice): $(B,T,C) \to (B,T,3C)$, split. Backward: matmul (#2) + split (#4).
2. **Split heads** — reshape to $(B,T,H,hs)$, transpose to $(B,H,T,hs)$.
   Backward: inverse transpose, inverse reshape (#4).
3. **Scores** — $S = QK^{\mathsf T}$: $(B,H,T,hs) \times (B,H,hs,T) \to (B,H,T,T)$.
   Backward, by #2:

   $$
   g_Q = g_S K,
   \qquad
   \boxed{\;g_K = g_S^{\mathsf T} Q.\;}
   $$

   > **Correction (caught in AI review, worth understanding for its own sake):**
   > the overnight write-up had $g_K = Q^{\mathsf T} g_S$. Shape-check it:
   > $(hs,T)\cdot(T,T) = (hs,T)$ — that is $K^{\mathsf T}$'s shape, not $K$'s.
   > $Q^{\mathsf T} g_S$ is the gradient **of $K^{\mathsf T}$**, the thing
   > literally multiplied. And the engine *does* compute it: at
   > [model.py:87](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/model.py#L87) the graph is `k.transpose(-2,-1)` then matmul,
   > so #2 produces $g_{K^{\mathsf T}} = Q^{\mathsf T} g_S$ at the transpose
   > node, and #4's transpose backward flips it: 
   > $g_K = (Q^{\mathsf T} g_S)^{\mathsf T} = g_S^{\mathsf T} Q$. Same quantity,
   > one op later — the error was a label, but the label is the difference
   > between the graph's midpoint and its endpoint.

4. **Scale** — $S' = S/\sqrt{hs}$. Backward: same constant (#0 mul).
5. **Causal mask** — $S'' = S' + M$, $M$ zero on/below the diagonal and a
   **large finite negative** ($-10^9$) above. Backward: add passes gradient
   through; $M$ is a constant with no gradient path. Why not $-\infty$? After
   softmax a masked position has $p = 0$, and the #7 backward multiplies by
   $p$ — with $-\infty$ the forward produces $e^{-\infty} = 0$ safely but mixed
   expressions like $0 \cdot \infty$ appear in float grad paths and produce
   NaN; $-10^9$ underflows to $p = 0$ with no infinities anywhere
   ([model.py:73](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/model.py#L73)).
6. **Softmax** — $A = \mathrm{softmax}(S'')$ row-wise. Backward: #7's VJP.
7. **Weighted sum** — $O = AV$: $(B,H,T,T)\times(B,H,T,hs) \to (B,H,T,hs)$.
   Backward, #2 again: $g_A = g_O V^{\mathsf T}$, $g_V = A^{\mathsf T} g_O$
   (both label-correct here — check the shapes and see why this one needed no
   fixing: $A^{\mathsf T} g_O$ lands on $(T,hs)$, which *is* $V$'s shape).
8. **Merge heads** — transpose, reshape back to $(B,T,C)$: inverse of step 2.
9. **Output projection** — $Y = OW_O$: matmul (#2).

No dedicated attention backward exists anywhere in the engine — steps 1–9 are
[model.py:79](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/model.py#L79)–[model.py:91](https://github.com/GreaseMonkeyIT/gpt2_no_torch/blob/main/model.py#L91), and the backward is the
topo sort visiting these nine VJPs in reverse. That is the whole design thesis
of the engine.

## References

1. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
2. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
3. T. Parr, J. Howard, *The Matrix Calculus You Need for Deep Learning*, 2018.
4. D. Hendrycks, K. Gimpel, *Gaussian Error Linear Units (GELUs)*, 2016.
5. D. Kingma, J. Ba, *Adam: A Method for Stochastic Optimization*, 2015.
6. I. Loshchilov, F. Hutter, *Decoupled Weight Decay Regularization*, 2019.
7. CS231n course notes (Stanford); NumPy broadcasting documentation; one Medium
   explainer used as a reference for #8's presentation.
