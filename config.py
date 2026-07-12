"""Training presets. cfg["model"] fields are passed to model.GPTConfig(**...).

vocab_size=None means "fill from data/<dir>/meta.pkl at runtime" (char data).
"""

PRESETS = {
    # single fixed batch -> loss ~0; proves engine + model + optimizer end to end
    "overfit": {
        "model": dict(n_layer=2, n_head=2, n_embd=128, block_size=64,
                      vocab_size=None, dropout=0.0),
        "data_dir": "data/shakespeare_char",
        "overfit_single_batch": True,
        "batch_size": 8, "grad_accum": 1,
        "max_iters": 400, "log_interval": 10, "eval_interval": 200, "eval_iters": 5,
        "lr": 1e-3, "min_lr": 1e-4, "warmup_iters": 20, "lr_decay_iters": 400,
        "out_dir": "out/overfit",
    },
    # minutes on CPU; samples should look Shakespeare-ish
    "shakespeare_char": {
        "model": dict(n_layer=4, n_head=4, n_embd=128, block_size=128,
                      vocab_size=None, dropout=0.0),
        "data_dir": "data/shakespeare_char",
        "overfit_single_batch": False,
        "batch_size": 16, "grad_accum": 1,
        "max_iters": 3000, "log_interval": 20, "eval_interval": 250, "eval_iters": 20,
        "lr": 1e-3, "min_lr": 1e-4, "warmup_iters": 100, "lr_decay_iters": 3000,
        "out_dir": "out/shakespeare_char",
    },
    # fallback if CuPy doesn't work out; max_iters is a ceiling — stop by wall clock
    "owt_cpu": {
        "model": dict(n_layer=4, n_head=4, n_embd=256, block_size=128,
                      vocab_size=50257, dropout=0.0),
        "data_dir": "data/openwebtext",
        "overfit_single_batch": False,
        "batch_size": 8, "grad_accum": 4,
        "max_iters": 20000, "log_interval": 10, "eval_interval": 500, "eval_iters": 20,
        "lr": 6e-4, "min_lr": 6e-5, "warmup_iters": 200, "lr_decay_iters": 20000,
        "out_dir": "out/owt_cpu",
    },
    # RTX 3050 Laptop (4GB) via CuPy (~30M params with embeddings)
    "owt_gpu": {
        "model": dict(n_layer=6, n_head=6, n_embd=384, block_size=256,
                      vocab_size=50257, dropout=0.0),
        "data_dir": "data/openwebtext",
        "device": "gpu",
        "overfit_single_batch": False,
        # microbatch 2: the (B*T, 50257) logits/CE buffers are the VRAM hogs and
        # 4GB WDDM silently pages past ~3GB (measured: 20x slowdown). accum 32
        # preserves 16K tokens per step. Run with CUPY_GPU_MEMORY_LIMIT=3GB so
        # overflow fails loudly instead of paging quietly.
        "batch_size": 2, "grad_accum": 32,
        # sized for the ~20h window at ~3.5K tok/s: 15K iters x 16K tok = 245M tokens
        "max_iters": 15000, "log_interval": 10, "eval_interval": 500, "eval_iters": 20,
        "lr": 6e-4, "min_lr": 6e-5, "warmup_iters": 500, "lr_decay_iters": 15000,
        "out_dir": "out/owt_gpu",
    },
}
