#!/usr/bin/env python3
"""Paper figure 05: cross-workload validation of the headline finding.

Renders a 2x2 grid (one panel per YCSB workload) of skew sweeps comparing
hybrid_staging vs cpu_only at the headline config:

  120 B + -rT-fF + 20 M + EPIC_YCSB_CACHE_CAP=6720000 (~33.6 % cache/DB ratio)

The cap matches the field-unit autosizer's natural 33.6 % pick at
-rT-fT/-rF-fT so plots 05-08 share one matched cache ratio (PLOTS.md
Appendix B), ensuring the staging machinery actually has to evict and admit
rather than holding the entire DB in cache. The 120 B vs 1 KB record-size
comparison itself lives in plot 03 (at its own 52.7 % convention); this
plot's purpose is to show the hybrid > cpu_only finding generalizes across
all four YCSB workloads, not just ycsbf.

gpu_only is intentionally absent here: at this cache pressure the question
"would you use gpu_only instead" is irrelevant; we're measuring the
hybrid-vs-cpu story specifically.

Reads $EGAD_LOGS_DIR/beyond_hbm/*.log; writes
$EGAD_FIGURES_DIR/beyond_hbm.{pdf,png,csv}.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType embedding for paper-quality PDFs
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"  # match the paper body text
matplotlib.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]  # Computer Modern Roman (LaTeX default)
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 14  # +2pt above matplotlib default 10  # avoid cmr10 missing-glyph warning
matplotlib.rcParams["mathtext.fontset"] = "cm"  # Computer Modern math (LaTeX-style)
matplotlib.rcParams["text.usetex"] = True  # render through real LaTeX (matches paper body)
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["xtick.direction"] = "in"
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
OUT_BASE = os.path.join(FIGURES_DIR, "beyond_hbm")

NTX_PER_EPOCH = 100_000
HYBRID_WINDOW = (250, 280)
BASELINE_WINDOW = (2, 5)

WORKLOAD_TITLES = {
    "ycsba": "YCSB-A",
    "ycsbb": "YCSB-B",
    "ycsbc": "YCSB-C",
    "ycsbf": "YCSB-F",
}
WORKLOAD_ORDER = ["ycsba", "ycsbb", "ycsbc", "ycsbf"]

# 120B = blue, 1KB = red. Solid = hybrid, dashed = cpu_only.
C_120B = "#1f4e79"
C_1KB  = "#b03a2e"

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\].*Running epoch (\d+)"
)
FILENAME_RE = re.compile(
    r"(?P<bench>ycsb[abcf])_(?P<size>120B|1KB)_(?P<mode>hybrid_staging|cpu_only)_skew(?P<skew>[\d.]+)_rep(?P<rep>\d+)\.log$"
)


def steady_state_mtxn(log_path: str, window: Tuple[int, int]) -> Optional[float]:
    times: Dict[int, datetime] = {}
    with open(log_path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                ts = datetime.strptime(
                    f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S.%f"
                )
                times[int(m.group(3))] = ts
    a, b = window
    if a not in times or b not in times:
        return None
    dt = (times[b] - times[a]).total_seconds()
    if dt <= 0:
        return None
    return ((b - a) * NTX_PER_EPOCH) / dt / 1e6


def collect() -> Dict[Tuple[str, str, str, float], List[float]]:
    by_cell: Dict[Tuple[str, str, str, float], List[float]] = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        m = FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        bench = m.group("bench")
        size = m.group("size")
        mode = m.group("mode")
        skew = float(m.group("skew"))
        window = HYBRID_WINDOW if mode == "hybrid_staging" else BASELINE_WINDOW
        mt = steady_state_mtxn(log, window)
        if mt is not None:
            by_cell[(bench, size, mode, skew)].append(mt)
    return by_cell


def render(cells: Dict[Tuple[str, str, str, float], List[float]]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    skews = sorted({k[3] for k in cells.keys()})
    if not skews:
        raise SystemExit("no cells parsed; did 05_beyond_hbm.sh run?")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0), sharex=True, sharey=True)

    series_cfg = [
        ("120B", "hybrid_staging", C_120B, "-",  "o", "EGAD"),
        ("120B", "cpu_only",       C_120B, "--", "x", "EPIC-CPU"),
    ]

    for ax, bench in zip(axes, WORKLOAD_ORDER):
        for size, mode, color, ls, marker, label in series_cfg:
            ys: List[Optional[float]] = []
            for s in skews:
                vs = cells.get((bench, size, mode, s), [])
                ys.append(statistics.median(vs) if vs else None)
            xs_plot = [s for s, y in zip(skews, ys) if y is not None]
            ys_plot = [y for y in ys if y is not None]
            if ys_plot:
                ax.plot(
                    xs_plot, ys_plot,
                    color=color, linestyle=ls, marker=marker, markersize=7,
                    linewidth=1.6, label=label,
                )
        ax.set_title(WORKLOAD_TITLES[bench], fontsize=14)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(axis="both", labelsize=12)

    for ax in axes:
        ax.set_xlabel(r"Zipfian skew $\theta$", fontsize=13)
    axes[0].set_ylabel("Throughput (MTxn/s)", fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.18),
        ncol=2, fontsize=13, frameon=False,
    )
    # fig.suptitle(
    #     "YCSB -rT-fF, 120 B, 20 M records, ~33.6 % cache/DB ratio: hybrid_staging vs cpu_only across four workloads",
    #     fontsize=10, y=1.06,
    # )
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "size", "mode", "skew", "median_mtxn_s", "n_reps"])
        for (bench, size, mode, skew), vs in sorted(cells.items()):
            w.writerow([bench, size, mode, skew, statistics.median(vs), len(vs)])
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(
            f"log dir not found: {LOG_DIR}; run 05_beyond_hbm.sh first"
        )
    cells = collect()
    render(cells)


if __name__ == "__main__":
    main()
