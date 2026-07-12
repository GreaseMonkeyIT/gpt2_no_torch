"""Loss + perplexity plots from a run's log.csv -> plots/<run>_{loss,ppl}.png.

Usage: .venv\\Scripts\\python.exe plots\\make_plots.py out\\shakespeare_char
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference dataviz palette (categorical slots 1-2 + chrome inks, light mode)
TRAIN, VAL = "#2a78d6", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                         "#898781", "#e1e0d9", "#c3c2b7")


def ema(x, span):
    out, m = np.empty_like(x), x[0]
    a = 2.0 / (span + 1)
    for i, v in enumerate(x):
        m = a * v + (1 - a) * m
        out[i] = m
    return out


def styled_axes(title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    return fig, ax


def plot(run_dir):
    run = os.path.basename(os.path.normpath(run_dir))
    it, tr, vit, vl = [], [], [], []
    with open(os.path.join(run_dir, "log.csv"), newline="") as f:
        for row in csv.DictReader(f):
            it.append(int(row["iter"]))
            tr.append(float(row["train_loss"]))
            if row["val_loss"]:
                vit.append(int(row["iter"]))
                vl.append(float(row["val_loss"]))
    it, tr = np.array(it), np.array(tr)
    vit, vl = np.array(vit), np.array(vl)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for kind, tf in (("loss", lambda y: y), ("ppl", np.exp)):
        ylab = "cross-entropy (nats)" if kind == "loss" else "perplexity"
        fig, ax = styled_axes(f"{run} — train/val {ylab.split(' ')[0]}", ylab)
        ax.plot(it, tf(tr), color=TRAIN, linewidth=0.9, alpha=0.35)
        ax.plot(it, tf(ema(tr, max(5, len(tr) // 20))), color=TRAIN,
                linewidth=2, label="train (smoothed)")
        if len(vit):
            ax.plot(vit, tf(vl), color=VAL, linewidth=2, marker="o",
                    markersize=5, label="val")
            ax.annotate(f"{tf(vl)[-1]:,.2f}", (vit[-1], tf(vl)[-1]),
                        xytext=(6, -3), textcoords="offset points",
                        color=INK2, fontsize=9)
        if kind == "ppl":
            ax.set_yscale("log")
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
        path = os.path.join(out_dir, f"{run}_{kind}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. out/shakespeare_char")
    args = ap.parse_args()
    plot(args.run_dir)
