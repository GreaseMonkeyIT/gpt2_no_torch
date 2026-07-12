"""Single entry point for every runnable script.

    python run.py <command> [args...]

Commands live in scripts/; the library they import (engine/, model.py, optim.py,
config.py, checkpoint.py) stays at the repo root. run.py puts the root on
sys.path and hands the remaining args to the chosen script, so everything runs
from the repo root regardless of where the scripts sit.
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

COMMANDS = {
    "train": "train.py",
    "sample": "sample.py",
    "status": "status.py",
    "watchdog": "watchdog.py",
    "eval": "eval_lm.py",
    "fullval": "full_val.py",
    "plot": "make_plots.py",
    "prep-shakespeare": "prepare_shakespeare.py",
    "prep-owt": "prepare_openwebtext.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: python run.py <command> [args...]")
        print("commands:", ", ".join(COMMANDS))
        sys.exit(0 if len(sys.argv) < 2 else 1)
    script = os.path.join(ROOT, "scripts", COMMANDS[sys.argv[1]])
    sys.argv = [script] + sys.argv[2:]           # the script sees its own argv
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
