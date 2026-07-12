"""Numpy autograd engine: micrograd's Value, generalized to arrays.

Same skeleton as I had previously proved in my warmup notebooks. Each op builds `out`
with a closure `_backward` that computes the VJP (given dL/dout, add dL/din
into each input's .grad), and then backward() runs the closures in reverse
topological order. 
The three upgrades over micrograd which I figured out might save time: 
-> values are np.ndarrays,
-> broadcasting gradients are un-summed (DERIVATIONS.md #1),
-> matmul/reduction/gather operations exist (#2-#8).

Convention: mixing a Tensor with an int/float/np.ndarray treats the latter as a
CONSTANT (no gradient).
This is used for the causal mask and scalar scaling.
datatype policy: float32 in training; gradient checks pass float64 through.

Array backend: `xp` is numpy (default) or cupy after use_gpu() — same code runs
on CPU and GPU because cupy mirrors numpy's API. The two exceptions are wrapped
below: scatter-add (cupyx) and device->host transfer (to_numpy).
"""

import math

import numpy as np

xp = np  # module-global array backend; every op looks it up at call time


def use_gpu():
    """Switch the engine to CuPy. To be called before building any Tensor/model (notebook suggestion)."""
    global xp
    import cupy
    xp = cupy


def to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _scatter_add(target, idx, src):
    if xp is np:
        np.add.at(target, idx, src)
    else:
        import cupyx
        cupyx.scatter_add(target, idx, src)


def unbroadcast(g, shape):
    """Sum g down to `shape`, undoing numpy broadcasting (#1).

    Broadcasting is implicit copying, and a value copied k times is a k-way
    fan-out, so its gradient is the sum over the copies."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)                    # summed over added leading axes
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)  # summed over stretched size-1 axes
    return g


def _accum(t, g):
    """Add a VJP contribution into t.grad (fan-out contributions sum)."""
    if t.grad is None:
        t.grad = xp.zeros_like(t.data)
    t.grad += g


class Tensor:
    __array_ufunc__ = None  # make `np_array + Tensor` fail loudly, not silently

    def __init__(self, data, requires_grad=False):
        self.data = xp.asarray(data)
        self.grad = None                      # np.ndarray like data, or None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = ()

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.data.dtype})"

    # ---- arithmetic (constants: int/float/np.ndarray get no gradient) ----

    def __add__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data)
            out._prev = (self, other)

            def _bw():
                _accum(self, unbroadcast(out.grad, self.data.shape))
                _accum(other, unbroadcast(out.grad, other.data.shape))
        else:
            out = Tensor(self.data + other)
            out._prev = (self,)

            def _bw():
                _accum(self, unbroadcast(out.grad, self.data.shape))
        out._backward = _bw
        return out

    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data)
            out._prev = (self, other)

            def _bw():
                _accum(self, unbroadcast(other.data * out.grad, self.data.shape))
                _accum(other, unbroadcast(self.data * out.grad, other.data.shape))
        else:
            out = Tensor(self.data * other)
            out._prev = (self,)

            def _bw():
                _accum(self, unbroadcast(other * out.grad, self.data.shape))
        out._backward = _bw
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -1.0 * other)

    def __truediv__(self, other):
        assert isinstance(other, (int, float)), "divide by constants only"
        return self * (1.0 / other)

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "constant exponents only"
        out = Tensor(self.data ** n)
        out._prev = (self,)

        def _bw():
            _accum(self, n * self.data ** (n - 1) * out.grad)
        out._backward = _bw
        return out

    def __matmul__(self, other):
        # C = A @ B; dA = g @ B^T, dB = A^T @ g (#2), transposing the last two
        # axes; unbroadcast sums batch dims when a weight was broadcast
        out = Tensor(self.data @ other.data)
        out._prev = (self, other)

        def _bw():
            g = out.grad
            _accum(self, unbroadcast(g @ other.data.swapaxes(-1, -2), self.data.shape))
            _accum(other, unbroadcast(self.data.swapaxes(-1, -2) @ g, other.data.shape))
        out._backward = _bw
        return out

    # ---- reductions (#3) ----

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims))
        out._prev = (self,)

        def _bw():
            g = out.grad
            if axis is not None and not keepdims:
                g = xp.expand_dims(g, axis)   # restore the collapsed axis…
            _accum(self, xp.broadcast_to(g, self.data.shape))  # …then spread
        out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        s = self.sum(axis=axis, keepdims=keepdims)
        return s * (float(s.data.size) / self.data.size)  # 1/n, n = collapsed count

    # ---- data movement: backward is the inverse rearrangement (#4) ----

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(shape))
        out._prev = (self,)

        def _bw():
            _accum(self, out.grad.reshape(self.data.shape))
        out._backward = _bw
        return out

    def transpose(self, ax1, ax2):
        out = Tensor(self.data.swapaxes(ax1, ax2))
        out._prev = (self,)

        def _bw():
            _accum(self, out.grad.swapaxes(ax1, ax2))
        out._backward = _bw
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx])
        out._prev = (self,)

        def _bw():
            g = xp.zeros_like(self.data)
            _scatter_add(g, idx, out.grad)
            _accum(self, g)
        out._backward = _bw
        return out

    # ---- elementwise nonlinearities (#0) ----

    def exp(self):
        out = Tensor(xp.exp(self.data))
        out._prev = (self,)

        def _bw():
            _accum(self, out.data * out.grad)
        out._backward = _bw
        return out

    def log(self):
        out = Tensor(xp.log(self.data))
        out._prev = (self,)

        def _bw():
            _accum(self, out.grad / self.data)
        out._backward = _bw
        return out

    def tanh(self):
        out = Tensor(xp.tanh(self.data))
        out._prev = (self,)

        def _bw():
            _accum(self, (1.0 - out.data ** 2) * out.grad)
        out._backward = _bw
        return out

    # ---- backprop driver: iterative topo sort, then closures in reverse ----

    def backward(self):
        assert self.data.size == 1, "backward() starts from a scalar loss"
        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                stack.append((child, False))
        self.grad = xp.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
            if node._prev:
                # intermediate node: consumers already ran (reverse topo) and
                # its own VJP just ran — free the activation-grad immediately.
                # Leaves (params, _prev=()) keep .grad for the optimizer.
                node.grad = None


# ---- fused ops ----

def gelu(x):
    """GPT-2's tanh-approximation GELU, fused for memory (#6)."""
    c = math.sqrt(2.0 / math.pi)
    xd = x.data
    t = xp.tanh(c * (xd + 0.044715 * xd ** 3))
    out = Tensor(0.5 * xd * (1.0 + t))
    out._prev = (x,)

    def _bw():
        # product rule on 0.5·x·(1+t)  +  chain rule through t(x)
        dt = (1.0 - t ** 2) * c * (1.0 + 3 * 0.044715 * xd ** 2)
        _accum(x, (0.5 * (1.0 + t) + 0.5 * xd * dt) * out.grad)
    out._backward = _bw
    return out


def softmax(x, axis=-1):
    """Row-wise softmax; max-subtraction is legal by shift invariance (#7)."""
    z = x.data - x.data.max(axis=axis, keepdims=True)
    e = xp.exp(z)
    p = e / e.sum(axis=axis, keepdims=True)
    out = Tensor(p)
    out._prev = (x,)

    def _bw():
        g = out.grad
        _accum(x, p * (g - (g * p).sum(axis=axis, keepdims=True)))
    out._backward = _bw
    return out


def softmax_cross_entropy(logits, targets):
    """Mean NLL of int targets under softmax(logits), fused (#8).

    logits: Tensor (..., V); targets: int array, shape = logits.shape[:-1].
    Returns a scalar Tensor. Backward is the famously simple fused form.
    """
    ld = logits.data
    V = ld.shape[-1]
    flat = ld.reshape(-1, V)
    y = xp.asarray(targets).reshape(-1)
    n = flat.shape[0]
    z = flat - flat.max(axis=1, keepdims=True)
    logp = z - xp.log(xp.exp(z).sum(axis=1, keepdims=True))   # log-softmax
    out = Tensor(xp.asarray(-logp[xp.arange(n), y].mean()))
    out._prev = (logits,)

    def _bw():
        d = xp.exp(logp)                  # softmax probabilities
        d[xp.arange(n), y] -= 1.0         # #8's result
        d /= n                            # the mean's 1/(B·T)
        _accum(logits, (d * out.grad).reshape(ld.shape))
    out._backward = _bw
    return out


def embedding(w, idx):
    """Row gather out[...] = w[idx[...]]; backward scatter-ADDS (#5) because a
    token id appearing k times in the batch is a k-way fan-out of its row."""
    idx = xp.asarray(idx)
    out = Tensor(w.data[idx])
    out._prev = (w,)

    def _bw():
        g = xp.zeros_like(w.data)
        _scatter_add(g, idx, out.grad)    # += per occurrence; assignment would drop repeats
        _accum(w, g)
    out._backward = _bw
    return out
