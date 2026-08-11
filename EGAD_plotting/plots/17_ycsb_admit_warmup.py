#!/usr/bin/env python3
"""Paper figure 17: per-epoch admission volume across the run, YCSB 20 M.

Motivates the steady-state measurement window. The cache starts empty,
so early epochs admit heavily; the per-epoch admission volume falls as
the hot set accumulates and flattens long before the e250-280 window
every YCSB throughput cell reports (shaded). One line per Zipfian skew,
each the mean across the three headline-configuration reps
(120 B records, 20 M rows, hybrid_staging, the test-05 logs).

Reads $EGAD_LOGS_DIR/beyond_hbm/ycsb?_120B_hybrid_staging_skew*_rep*.log;
writes $EGAD_FIGURES_DIR/ycsb_admit_warmup.{pdf,png,csv}.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["axes.formatter.use_mathtext"] = True
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["xtick.direction"] = "out"
matplotlib.rcParams["ytick.direction"] = "in"
import matplotlib.pyplot as plt

LOGS_DIR = os.environ.get(
    "EGAD_LOGS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"),
)
FIGURES_DIR = os.environ.get(
    "EGAD_FIGURES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"),
)
os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_DIR = os.path.join(LOGS_DIR, "beyond_hbm")
OUT_BASE = os.path.join(FIGURES_DIR, "ycsb_admit_warmup")

BENCH = "ycsbf"  # the workload the setup section's window discussion uses
FILENAME_RE = re.compile(
    rf"{BENCH}_120B_hybrid_staging_skew(?P<skew>[0-9.]+)_rep(?P<rep>\d+)\.log$"
)
RUNNING_RE = re.compile(r"Running epoch (\d+)")
ADMIT_RE = re.compile(r"new=(\d+) split: miss-admit=(\d+)")

WINDOW = (250, 280)


def parse_admits(path: str) -> Dict[int, int]:
    """epoch -> admitted records (0 when the log prints no admit line)."""
    admits: Dict[int, int] = {}
    cur = None
    with open(path, errors="replace") as f:
        for line in f:
            m = RUNNING_RE.search(line)
            if m:
                cur = int(m.group(1))
                admits.setdefault(cur, 0)
                continue
            m = ADMIT_RE.search(line)
            if m and cur is not None:
                admits[cur] = admits.get(cur, 0) + int(m.group(1))
    return admits


def collect() -> Dict[float, Dict[int, List[int]]]:
    by_skew: Dict[float, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        m = FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        skew = float(m.group("skew"))
        for epoch, n in parse_admits(log).items():
            by_skew[skew][epoch].append(n)
    return by_skew


def render(by_skew: Dict[float, Dict[int, List[int]]]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)
    if not by_skew:
        raise SystemExit(f"no admit series parsed under {LOG_DIR}; "
                         "run 05_beyond_hbm.sh first")

    skews = sorted(by_skew.keys())
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    rows = []
    for i, s in enumerate(skews):
        series = by_skew[s]
        epochs = sorted(e for e in series if e >= 1)
        ys = [statistics.mean(series[e]) / 1e6 for e in epochs]
        ax.plot(epochs, ys, color=cmap(0.1 + 0.8 * i / max(1, len(skews) - 1)),
                linewidth=1.4, label=rf"$\theta = {s:g}$")
        for e, y in zip(epochs, ys):
            rows.append([s, e, round(y, 4)])

    ax.axvspan(WINDOW[0], WINDOW[1], color="0.85", alpha=0.6, zorder=0)
    ax.annotate("window", xy=(WINDOW[0] + 3, ax.get_ylim()[1] * 0.20),
                fontsize=11, color="0.35", rotation=90, va="center")
    ax.set_xlabel("epoch", fontsize=13)
    ax.set_ylabel("admitted records (M)", fontsize=13)
    ax.set_xlim(0, 300)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="upper right", fontsize=11, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    with open(OUT_BASE + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["skew", "epoch", "admitted_records_m_mean"])
        w.writerows(rows)
    print(f"saved {OUT_BASE}.csv")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(f"log dir not found: {LOG_DIR}; run 05_beyond_hbm.sh first")
    render(collect())


if __name__ == "__main__":
    main()
