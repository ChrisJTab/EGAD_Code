#!/usr/bin/env python3
"""Paper figure 13: combined cross-workload skew sweep, 4 lines per panel.

One row, 4 panels (one per YCSB workload). Each panel shows 4 lines
spanning BOTH plot 04's regime (fits in HBM, 5 M records) AND plot 05's
regime (exceeds HBM, 20 M records):

  red  solid  = hybrid_staging at  5 M records (fits in HBM)
  red  dashed = gpu_only       at  5 M records (the in-HBM ceiling)
  blue solid  = hybrid_staging at 20 M records (exceeds HBM)
  blue dashed = cpu_only       at 20 M records (the exceeds-HBM floor)

Reads from BOTH plot 04's three_way/ logs and plot 05's beyond_hbm/
logs. Equivalent to plots 04+05 stacked into one figure with shared
axes. No new benchmark runs needed.

Writes $EGAD_FIGURES_DIR/workload_sweep_combined.{pdf,png,csv}.
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
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["xtick.direction"] = "out"  # paper-style: ticks point outward, label sits lower
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

TW_DIR = os.path.join(LOGS_DIR, "three_way")
BH_DIR = os.path.join(LOGS_DIR, "beyond_hbm")
OUT_BASE = os.path.join(FIGURES_DIR, "workload_sweep_combined")

NTX_PER_EPOCH = 100_000
HYBRID_WINDOW = (250, 280)
BASELINE_WINDOW = (2, 5)   # gpu_only: 5-epoch run (no warmup needed on GPU-resident baseline)
CPU_WINDOW = (3, 10)       # cpu_only: average over epochs 3-10 (steady-state, post-warmup)

WORKLOAD_TITLES = {
    "ycsba": "YCSB-A",
    "ycsbb": "YCSB-B",
    "ycsbc": "YCSB-C",
    "ycsbf": "YCSB-F",
}
WORKLOAD_ORDER = ["ycsba", "ycsbb", "ycsbc", "ycsbf"]

C_5M  = "#b03a2e"  # red:  5 M records (fits in HBM)
C_20M = "#1f4e79"  # blue: 20 M records (exceeds HBM)

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\].*Running epoch (\d+)"
)
TW_FILENAME_RE = re.compile(
    r"(?P<bench>ycsb[abcf])_(?P<mode>hybrid_staging|gpu_only)_skew(?P<skew>[\d.]+)_rep(?P<rep>\d+)\.log$"
)
BH_FILENAME_RE = re.compile(
    r"(?P<bench>ycsb[abcf])_120B_(?P<mode>hybrid_staging|cpu_only)_skew(?P<skew>[\d.]+)_rep(?P<rep>\d+)\.log$"
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


def collect_dir(dir_path: str, fn_re: re.Pattern) -> Dict[Tuple[str, str, float], List[float]]:
    by_cell: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(dir_path, "*.log"))):
        m = fn_re.search(os.path.basename(log))
        if not m:
            continue
        bench = m.group("bench")
        mode = m.group("mode")
        skew = float(m.group("skew"))
        if mode == "hybrid_staging":
            window = HYBRID_WINDOW
        elif mode == "cpu_only":
            window = CPU_WINDOW
        else:
            window = BASELINE_WINDOW
        mt = steady_state_mtxn(log, window)
        if mt is not None:
            by_cell[(bench, mode, skew)].append(mt)
    return by_cell


def render(top_cells, bot_cells) -> None:
    """top_cells = three_way (5 M); bot_cells = beyond_hbm (20 M)."""
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    skews = sorted({k[2] for k in top_cells.keys()} | {k[2] for k in bot_cells.keys()})
    if not skews:
        raise SystemExit("no cells parsed; did 04_three_way.sh and 05_beyond_hbm.sh run?")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0), sharex=True, sharey=True)

    # series_specs: (source_cells, mode, color, linestyle, marker, label)
    series_specs = [
        (top_cells, "hybrid_staging", C_5M,  "-",  "o", "EGAD, 5 M"),
        (top_cells, "gpu_only",       C_5M,  "--", "x", "EPIC-GPU, 5 M"),
        (bot_cells, "hybrid_staging", C_20M, "-",  "o", "EGAD, 20 M"),
        (bot_cells, "cpu_only",       C_20M, "--", "x", "EPIC-CPU, 20 M"),
    ]

    for ax, bench in zip(axes, WORKLOAD_ORDER):
        for cells, mode, color, ls, marker, label in series_specs:
            pts = []  # (skew, mean, min, max) -- center stays the mean
            for s in skews:
                vs = cells.get((bench, mode, s), [])
                if vs:
                    pts.append((s, statistics.mean(vs), min(vs), max(vs)))
            if pts:
                xs_plot = [p[0] for p in pts]
                ys_plot = [p[1] for p in pts]
                yerr = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
                ax.errorbar(
                    xs_plot, ys_plot, yerr=yerr, capsize=2.5, elinewidth=1.0,
                    color=color, linestyle=ls, marker=marker, markersize=7,
                    linewidth=1.6, label=label,
                )
        ax.set_title(WORKLOAD_TITLES[bench], fontsize=14)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(axis="both", labelsize=12)

    # Anchor at 0 AFTER all data is plotted so sharey autoscale captures
    # the global max across all panels before we lock the bottom edge.
    axes[0].set_xlim(left=0)
    axes[0].set_ylim(bottom=0)

    for ax in axes:
        ax.set_xlabel(r"Zipfian skew $\theta$", fontsize=13)
    axes[0].set_ylabel("Throughput (MTxn/s)", fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.18),
        ncol=4, fontsize=12, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["records", "bench", "mode", "skew", "mean_mtxn_s", "min_mtxn_s", "max_mtxn_s", "n_reps"])
        for label, cells in [("5M", top_cells), ("20M", bot_cells)]:
            for (bench, mode, skew), vs in sorted(cells.items()):
                w.writerow([label, bench, mode, skew, round(statistics.mean(vs), 3),
                            round(min(vs), 3), round(max(vs), 3), len(vs)])
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(TW_DIR) or not os.path.isdir(BH_DIR):
        raise SystemExit(f"missing source log dirs: {TW_DIR}, {BH_DIR}")
    top = collect_dir(TW_DIR, TW_FILENAME_RE)
    bot = collect_dir(BH_DIR, BH_FILENAME_RE)
    render(top, bot)


if __name__ == "__main__":
    main()
