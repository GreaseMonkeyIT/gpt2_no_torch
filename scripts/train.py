"""Training loop plumbing. The math lives in engine/, model.py, optim.py.

Usage (from repo root):
  .venv\\Scripts\\python.exe train.py shakespeare_char
  .venv\\Scripts\\python.exe train.py owt_gpu --resume

Ctrl+C is safe: saves a checkpoint before exiting.
"""

import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import checkpoint
from config import PRESETS


def get_batch(data, block_size, batch_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=PRESETS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-iters", type=int, default=None,
                    help="override the preset's max_iters")
    args = ap.parse_args()
    cfg = dict(PRESETS[args.preset])
    if args.max_iters is not None:
        cfg["max_iters"] = args.max_iters

    from engine import tensor as et
    if cfg.get("device") == "gpu":
        et.use_gpu()   # must happen before any Tensor/model is built
        print("device: gpu (cupy)")

    import optim as optim_mod
    from model import GPT, GPTConfig
    from optim import AdamW, get_lr

    train_data = np.memmap(os.path.join(cfg["data_dir"], "train.bin"),
                           dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(cfg["data_dir"], "val.bin"),
                         dtype=np.uint16, mode="r")

    mcfg = dict(cfg["model"])
    if mcfg["vocab_size"] is None:
        with open(os.path.join(cfg["data_dir"], "meta.pkl"), "rb") as f:
            mcfg["vocab_size"] = pickle.load(f)["vocab_size"]

    model = GPT(GPTConfig(**mcfg))
    params = model.parameters()
    n_params = sum(p.data.size for p in params)
    print(f"{args.preset}: {n_params / 1e6:.2f}M params, vocab {mcfg['vocab_size']}")

    opt = AdamW(params, lr=cfg["lr"])
    clip = getattr(optim_mod, "clip_grad_norm", None)
    start_iter = 0

    os.makedirs(cfg["out_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["out_dir"], "ckpt.npz")
    log_path = os.path.join(cfg["out_dir"], "log.csv")

    if args.resume:
        _, arrays, start_iter, opt_state = checkpoint.load(ckpt_path)
        for p, a in zip(params, arrays):
            p.data[...] = et.xp.asarray(a)
        if opt_state and hasattr(opt, "load_state_dict"):
            opt.load_state_dict(opt_state)
        print(f"resumed from iter {start_iter}")
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "train_loss", "val_loss", "lr", "tok_per_sec"])

    def save(it):
        opt_state = opt.state_dict() if hasattr(opt, "state_dict") else None
        checkpoint.save(ckpt_path, params,
                        {"model": mcfg, "data_dir": cfg["data_dir"]}, it, opt_state)

    rng = np.random.default_rng(1337)
    T, B = mcfg["block_size"], cfg["batch_size"]
    fixed = get_batch(train_data, T, B, rng) if cfg["overfit_single_batch"] else None
    tokens_per_iter = B * cfg["grad_accum"] * T
    t_last, it_last = time.time(), start_iter

    it = start_iter
    try:
        for it in range(start_iter, cfg["max_iters"]):
            lr = get_lr(it, cfg)
            opt.zero_grad()
            train_loss = 0.0
            for _ in range(cfg["grad_accum"]):
                x, y = fixed if fixed is not None else get_batch(train_data, T, B, rng)
                _, loss = model(x, y)
                loss.backward()
                train_loss += float(loss.data) / cfg["grad_accum"]
            if cfg["grad_accum"] > 1:
                for p in params:
                    if p.grad is not None:
                        p.grad /= cfg["grad_accum"]
            if clip is not None:
                clip(params, 1.0)
            opt.step(lr)

            last = it == cfg["max_iters"] - 1
            val_loss = ""
            if it % cfg["eval_interval"] == 0 or last:
                vl = float(np.mean([float(model(*get_batch(val_data, T, B, rng))[1].data)
                                    for _ in range(cfg["eval_iters"])]))
                val_loss = f"{vl:.4f}"
                save(it + 1)
                print(f"iter {it}: val_loss {vl:.4f} (ppl {np.exp(vl):.1f}) — checkpoint saved")

            if it % cfg["log_interval"] == 0 or last:
                now = time.time()
                tps = tokens_per_iter * max(it - it_last, 1) / (now - t_last)
                t_last, it_last = now, it
                print(f"iter {it}: loss {train_loss:.4f}, lr {lr:.2e}, {tps:,.0f} tok/s")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([it, f"{train_loss:.4f}", val_loss,
                                            f"{lr:.6f}", f"{tps:.0f}"])
    except KeyboardInterrupt:
        print(f"\ninterrupted at iter {it} — saving checkpoint")
    save(it + 1)


if __name__ == "__main__":
    main()
