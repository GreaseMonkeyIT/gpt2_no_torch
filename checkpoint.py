"""Checkpoint I/O. Relies on model.parameters() returning a deterministic order."""

import json

import numpy as np

from engine.tensor import to_numpy


def save(path, params, config, it, opt_state=None):
    arrays = {f"p{i}": to_numpy(p.data) for i, p in enumerate(params)}
    for k, v in (opt_state or {}).items():
        arrays[f"opt_{k}"] = to_numpy(v)
    np.savez(path, _config=json.dumps(config), _iter=it,
             _n_params=len(params), **arrays)


def load(path):
    with np.load(path) as z:   # close the handle — Windows blocks open files
        config = json.loads(str(z["_config"]))
        params = [z[f"p{i}"] for i in range(int(z["_n_params"]))]
        opt_state = {k[4:]: z[k] for k in z.files if k.startswith("opt_")}
        return config, params, int(z["_iter"]), opt_state
