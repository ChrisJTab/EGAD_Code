#!/usr/bin/env python3
"""Paper figure 09: TPCC warehouse-count sweep, three modes.

X-axis: number of warehouses (log-2 scale; tpccdeck W ∈ {4..128},
tpcc/NP W ∈ {4..240}). Y-axis: steady-state e30-e50 throughput in
MTxn/s (median over reps). Three lines:

  * hybrid_staging — solid blue
  * gpu_only       — dashed green ceiling (stock OFF layout; on the NP
                     panel the line ends at its true HBM ceiling W=208,
                     OOM from W=216 on the 32 GB card)
  * cpu_only       — dashed red floor

The headline pattern: hybrid_staging beats cpu_only from W=8 up on both
mixes, sits below gpu_only wherever gpu_only fits, and on NP continues
past gpu_only's ceiling (W=216, 240) still above cpu_only. This is the
Epic Fig 4 + Fig 8 analog.

Reads $EGAD_LOGS_DIR/tpcc_warehouse_sweep/*.log; writes
$EGAD_FIGURES_DIR/tpcc_warehouse_sweep.{pdf,png,csv}.
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

LOG_DIR = os.path.join(LOGS_DIR, "tpcc_warehouse_sweep")
OUT_BASE = os.path.join(FIGURES_DIR, "tpcc_warehouse_sweep")

NTX_PER_EPOCH = 100_000
WINDOW = (30, 50)

MODE_STYLE = {
    "hybrid_staging": dict(color="#1f4e79", linestyle="-",  marker="o",
                           label="EGAD"),
    "gpu_only":       dict(color="#2e7d32", linestyle="--", marker="s",
                           label="EPIC-GPU"),
    "cpu_only":       dict(color="#b03a2e", linestyle="--", marker="^",
                           label="EPIC-CPU"),
}
MODE_ORDER = ["hybrid_staging", "gpu_only", "cpu_only"]

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\].*Running epoch (\d+)"
)
FILENAME_RE = re.compile(
    r"(?P<mix>tpccdeck|tpcc)_w(?P<w>\d+)_(?P<mode>hybrid_staging|cpu_only|gpu_only)_rep(?P<rep>\d+)\.log$"
)
MIX_ORDER = ["tpccdeck", "tpcc"]
MIX_TITLES = {"tpccdeck": "TPC-C", "tpcc": "TPC-C NP"}


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


def collect() -> Dict[Tuple[str, str], Dict[int, List[float]]]:
    by_cell: Dict[Tuple[str, str], Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        m = FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        mix = m.group("mix")
        w = int(m.group("w"))
        mode = m.group("mode")
        mt = steady_state_mtxn(log, WINDOW)
        if mt is not None:
            by_cell[(mix, mode)][w].append(mt)
    return by_cell


def render(by_cell: Dict[Tuple[str, str], Dict[int, List[float]]]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    all_w = sorted({w for cells in by_cell.values() for w in cells})
    if not all_w:
        raise SystemExit("no cells parsed; did 09_tpcc_warehouse_sweep.sh run?")

    # NP extends past 128; label a thinned subset there so the log-2 axis
    # stays readable (160/208/216 remain unlabeled data points).
    NP_TICKS = [4, 8, 16, 32, 64, 128, 192, 240]

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 3.0), sharey=True)

    for ax, mix in zip(axes, MIX_ORDER):
        for mode in MODE_ORDER:
            cells = by_cell.get((mix, mode), {})
            if not cells:
                continue
            ws = sorted(cells.keys())
            ys = [statistics.median(cells[w]) for w in ws]
            yerr = [[y - min(cells[w]) for w, y in zip(ws, ys)],
                    [max(cells[w]) - y for w, y in zip(ws, ys)]]
            ax.errorbar(ws, ys, yerr=yerr, capsize=2.5, elinewidth=1.0,
                        markersize=9, linewidth=1.6, **MODE_STYLE[mode])
        mix_ws = sorted({w for mode in MODE_ORDER for w in by_cell.get((mix, mode), {})})
        ticks = [w for w in mix_ws if mix != "tpcc" or w in NP_TICKS]
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(w) for w in ticks])
        ax.set_xlabel(r"warehouses ($W$)", fontsize=13)
        ax.set_title(MIX_TITLES[mix], fontsize=14)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(axis="both", labelsize=12)

    # Anchor Y at 0 AFTER plotting so the shared autoscale has captured
    # the global max across both mixes (X stays log-2 so cannot include 0).
    axes[0].set_ylim(bottom=0)

    axes[0].set_ylabel("Throughput (MTxn/s)", fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.10),
        ncol=3, fontsize=13, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mix", "mode", "warehouses", "median_mtxn_s", "min_mtxn_s", "max_mtxn_s", "n_reps"])
        for mix in MIX_ORDER:
            for mode in MODE_ORDER:
                cells = by_cell.get((mix, mode), {})
                for ww in sorted(cells.keys()):
                    vs = cells[ww]
                    w.writerow([mix, mode, ww, round(statistics.median(vs), 3),
                                round(min(vs), 3), round(max(vs), 3), len(vs)])
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(
            f"log dir not found: {LOG_DIR}; run 09_tpcc_warehouse_sweep.sh first"
        )
    by_cell = collect()
    render(by_cell)


if __name__ == "__main__":
    main()
