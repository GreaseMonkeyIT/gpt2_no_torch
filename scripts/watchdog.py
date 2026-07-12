"""Overnight watchdog: keep the OWT GPU run alive, resuming from checkpoint.

The 3GB CuPy pool cap makes VRAM overflow fail loudly (OOM) instead of paging
over PCIe at a measured 20x slowdown; if training dies for any reason, relaunch
with --resume, at most 5 restarts so a systematic failure can't loop all night.
"""

import os
import subprocess
import time

os.environ["CUPY_GPU_MEMORY_LIMIT"] = "3221225472"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
OUT = os.path.join(ROOT, "out", "owt_gpu")
LOG = os.path.join(OUT, "watchdog.log")
os.makedirs(OUT, exist_ok=True)

for attempt in range(6):
    args = [PY, "-u", os.path.join(ROOT, "scripts", "train.py"), "owt_gpu"]
    if attempt > 0 and os.path.exists(os.path.join(OUT, "ckpt.npz")):
        args.append("--resume")
    with open(LOG, "a") as f:
        f.write(f"=== attempt {attempt}: {' '.join(args[2:])} ===\n")
        f.flush()
        rc = subprocess.call(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"=== exited rc={rc} ===\n")
    if rc == 0:
        break
    time.sleep(20)
