#!/usr/bin/env python3
"""Paper record-size limits figure.

Renders the record-size sensitivity story as a single 4-line skew sweep:

  120B hybrid_staging  solid blue   - wins big across all skews
  120B cpu_only        dashed blue  - 120B CPU baseline
  1KB  hybrid_staging  solid red    - drops sharply at low skew
  1KB  cpu_only        dashed red   - 1KB CPU baseline

The "showing its limits" payoff is the solid-red curve crossing under the
dashed-red one at low skew, while the solid-blue stays far above its
dashed counterpart. This is one plot that tells three stories at once:
hybrid's lead is record-size-dependent, 120B has a bandwidth advantage,
and CPU is roughly insensitive to record size on this workload.

Reads $EGAD_LOGS_DIR/record_size/*.log; writes
$EGAD_FIGURES_DIR/record_size.{pdf,png}.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
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

LOG_DIR = os.path.join(LOGS_DIR, "record_size")
OUT_BASE = os.path.join(FIGURES_DIR, "record_size")

NTX_PER_EPOCH = 100_000
HYBRID_WINDOW = (250, 280)
BASELINE_WINDOW = (3, 10)  # cpu_only: average over epochs 3-10 (steady-state, post-warmup)

# Steady-state palette (paper convention):
#   120B = blue family, 1KB = red family.
#   solid = hybrid_staging, dashed = cpu_only.
C_120B = "#1f4e79"
C_1KB  = "#b03a2e"

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\].*Running epoch (\d+)"
)
FILENAME_RE = re.compile(
    r"(?P<size>120B|1KB)_(?P<mode>hybrid_staging|cpu_only)_skew(?P<skew>[\d.]+)_rep(?P<rep>\d+)\.log$"
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


def collect() -> Dict[Tuple[str, str, float], List[float]]:
    """Return {(size, mode, skew): [throughputs]} medians-ready map."""
    by_cell: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        m = FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        size = m.group("size")
        mode = m.group("mode")
        skew = float(m.group("skew"))
        window = HYBRID_WINDOW if mode == "hybrid_staging" else BASELINE_WINDOW
        mt = steady_state_mtxn(log, window)
        if mt is not None:
            by_cell[(size, mode, skew)].append(mt)
    return by_cell


def render(cells: Dict[Tuple[str, str, float], List[float]]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    skews = sorted({k[2] for k in cells.keys()})
    if not skews:
        raise SystemExit("no cells parsed; did 03_record_size.sh run?")

    fig = plt.figure(figsize=(8.5, 5.2))
    ax = fig.add_axes([0.11, 0.13, 0.86, 0.70])

    series_cfg = [
        ("120B", "hybrid_staging", C_120B, "-",  "o", "120 B EGAD"),
        ("120B", "cpu_only",       C_120B, "--", "x", r"120 B EPIC-CPU"),
        ("1KB",  "hybrid_staging", C_1KB,  "-",  "o", "1 KB EGAD"),
        ("1KB",  "cpu_only",       C_1KB,  "--", "x", r"1 KB EPIC-CPU"),
    ]

    for size, mode, color, ls, marker, label in series_cfg:
        pts = []  # (skew, mean, min, max) -- center stays the mean
        for s in skews:
            vs = cells.get((size, mode, s), [])
            if vs:
                pts.append((s, statistics.mean(vs), min(vs), max(vs)))
        if not pts:
            continue
        xs_plot = [p[0] for p in pts]
        ys_plot = [p[1] for p in pts]
        yerr = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
        ax.errorbar(
            xs_plot, ys_plot, yerr=yerr, capsize=2.5, elinewidth=1.0,
            color=color, linestyle=ls, marker=marker, markersize=8,
            linewidth=1.6, label=label,
        )

    ax.set_xlabel(r"Zipfian skew $\theta$", fontsize=17)
    ax.set_ylabel("Throughput (MTxn/s)", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=4, fontsize=16, frameon=False)
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    # Also dump the per-cell median CSV for paper-text reference.
    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "mode", "skew", "mean_mtxn_s", "min_mtxn_s", "max_mtxn_s", "n_reps"])
        for (size, mode, skew), vs in sorted(cells.items()):
            w.writerow([size, mode, skew, statistics.mean(vs),
                        round(min(vs), 3), round(max(vs), 3), len(vs)])
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(
            f"log dir not found: {LOG_DIR}; run 03_record_size.sh first"
        )
    cells = collect()
    render(cells)


if __name__ == "__main__":
    main()
