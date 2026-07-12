"""Gradient checking: analytic (your engine) vs central finite differences.

Every op must pass this before it goes anywhere near the model.
- run checks in float64: h=1e-5 round-off is invisible at double precision
- central differences have O(h^2) truncation error (Taylor); tol=1e-6 relative
  is comfortable for smooth ops. Ops with kinks (max/relu) can trip the checker
  exactly at the kink — reroll the seed if a FAIL looks like that.

Self-test (no engine needed): python tests/grad_check.py
"""

import numpy as np


def numeric_grad(f, arrays, i, h=1e-5):
    """d f(arrays) / d arrays[i] by central differences. f returns a float."""
    x = arrays[i]
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        j = it.multi_index
        old = x[j]
        x[j] = old + h
        fp = f(*arrays)
        x[j] = old - h
        fm = f(*arrays)
        x[j] = old
        g[j] = (fp - fm) / (2 * h)
    return g


def check(f_engine, shapes, name="", seed=0, h=1e-5, tol=1e-6):
    """Check f_engine's backward against numeric gradients.

    f_engine: takes len(shapes) engine Tensors, returns a SCALAR Tensor.
              (Implement sum() early — reductions make everything checkable.)
    shapes:   shapes of the float inputs to perturb. Integer inputs (embedding
              indices, targets) should be captured in the closure, not passed here.
    """
    from engine.tensor import Tensor

    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(s) for s in shapes]

    ts = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    f_engine(*ts).backward()

    def f_np(*arrs):
        return float(f_engine(*[Tensor(a.copy()) for a in arrs]).data)

    ok = True
    for i, t in enumerate(ts):
        num = numeric_grad(f_np, [a.copy() for a in arrays], i, h)
        ana = t.grad
        assert ana is not None, f"{name} input{i}: grad is None"
        assert ana.shape == num.shape, \
            f"{name} input{i}: grad shape {ana.shape} != input shape {num.shape}"
        rel = np.abs(ana - num) / np.maximum(np.abs(ana) + np.abs(num), 1e-8)
        max_rel = float(rel.max()) if rel.size else 0.0
        passed = max_rel < tol
        ok &= passed
        print(f"[{'OK  ' if passed else 'FAIL'}] {name} input{i}: max_rel_err={max_rel:.2e}")
    return ok


if __name__ == "__main__":
    # harness self-test on pure numpy: f(x) = sum(x^2), analytic grad = 2x
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3))
    num = numeric_grad(lambda a: float((a ** 2).sum()), [x], 0)
    err = np.abs(num - 2 * x).max()
    print(f"self-test max_abs_err={err:.2e} ({'OK' if err < 1e-8 else 'FAIL'})")
