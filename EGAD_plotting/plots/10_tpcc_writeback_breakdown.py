#!/usr/bin/env python3
"""Paper figure 10: TPCC per-phase epoch breakdown, async vs sync.

The TPCC analog of plot 08 (YCSB writeback breakdown). Two stacked
horizontal bars at W=128 tpccdeck: async (-z true) and sync (-z false).
Each bar decomposed into the main-thread phases:

  index_transfer, indexing, submission, prepareEpoch (8-stager
  aggregate), initialization, execution, writeback

In TPCC the 8 per-table stagers run prepareEpoch concurrently on
per-stager streams under EPIC_PREPARE_EPOCH_PARALLEL=1, so the
"prepareEpoch" block in the bar is the WALLCLOCK between Phase 2
start and Phase 2 end (i.e., the slowest stager defines the block).

The "writeback" block is what async hides:
  sync mode  -> the full flush wall-clock (Flush phase on main thread)
  async mode -> flush_sync_prev + flush_start_async (the slivers on
                the main thread; the real D2H/scatter runs concurrently
                on per-stager worker threads, invisible here)

Reads $EGAD_LOGS_DIR/tpcc_writeback_breakdown/*.log; writes
$EGAD_FIGURES_DIR/tpcc_writeback_breakdown.{pdf,png,csv}.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import statistics
from collections import defaultdict
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
import matplotlib.patches as mpatches

LOGS_DIR = os.environ.get(
    "EGAD_LOGS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"),
)
FIGURES_DIR = os.environ.get(
    "EGAD_FIGURES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"),
)
os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_DIR = os.path.join(LOGS_DIR, "tpcc_writeback_breakdown")
OUT_BASE = os.path.join(FIGURES_DIR, "tpcc_writeback_breakdown")

# Steady-state window for TPCC (the headline e30-50 window).
EPOCH_LO, EPOCH_HI = 30, 50

PHASE_COLORS = {
    "index_transfer": "#6B9F71",
    "indexing":       "#93B883",
    "submission":     "#C9A961",
    "prepareEpoch":   "#1F4E79",   # the staging block
    "initialization": "#B8A8D5",
    "execution":      "#6B5390",
    "writeback":      "#B23A48",
}
PHASE_LABELS = {
    "index_transfer": "idx tsf",
    "indexing":       "indexing",
    "submission":     "submit",
    "prepareEpoch":   "stage (8-stager)",
    "initialization": "init",
    "execution":      "exec",
    "writeback":      "writeback",
}
PHASE_ORDER = [
    "index_transfer", "indexing", "submission",
    "prepareEpoch",
    "initialization", "execution",
    "writeback",
]

EPOCH_PHASE_RE = re.compile(
    r"Epoch (\d+) (index_transfer|indexing|submission|initialization|execution|"
    r"hybrid_staging:prepareEpoch|hybrid_staging:remap|"
    r"hybrid_staging:flush_sync_prev|hybrid_staging:flush_start_async|"
    r"hybrid_staging:periodicFlush) time: (\d+) us"
)
FILENAME_RE = re.compile(
    r"tpccdeck_w(?P<w>\d+)_(?P<mode>async|sync)_rep(?P<rep>\d+)\.log$"
)


def parse_per_epoch(log_path: str) -> Dict[str, float]:
    """Return median-per-phase across EPOCH_LO..EPOCH_HI for this log."""
    per_epoch: Dict[int, Dict[str, int]] = defaultdict(dict)
    with open(log_path) as f:
        for line in f:
            m = EPOCH_PHASE_RE.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            phase = m.group(2)
            dur = int(m.group(3))
            # Map phase markers to our render buckets.
            if phase == "hybrid_staging:prepareEpoch":
                per_epoch[ep]["prepareEpoch"] = per_epoch[ep].get("prepareEpoch", 0) + dur
            elif phase in ("hybrid_staging:flush_sync_prev",
                           "hybrid_staging:flush_start_async",
                           "hybrid_staging:periodicFlush"):
                per_epoch[ep]["writeback"] = per_epoch[ep].get("writeback", 0) + dur
            elif phase == "hybrid_staging:remap":
                # Tiny phase, merge into prepareEpoch for the bar (already
                # the dominant staging cost is in prepareEpoch).
                per_epoch[ep]["prepareEpoch"] = per_epoch[ep].get("prepareEpoch", 0) + dur
            else:
                per_epoch[ep][phase] = dur

    medians: Dict[str, List[int]] = defaultdict(list)
    for ep, ph in per_epoch.items():
        if EPOCH_LO <= ep <= EPOCH_HI:
            for k, v in ph.items():
                medians[k].append(v)
    return {k: statistics.median(v) for k, v in medians.items() if v}


def collect() -> Dict[str, Dict[str, float]]:
    """Return {mode: {phase: median_us}}."""
    per_cell: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        m = FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        mode = m.group("mode")
        per_rep = parse_per_epoch(log)
        if per_rep:
            per_cell[mode].append(per_rep)

    out: Dict[str, Dict[str, float]] = {}
    for mode, reps in per_cell.items():
        all_phases = set().union(*(r.keys() for r in reps))
        out[mode] = {
            phase: statistics.median([r.get(phase, 0) for r in reps])
            for phase in all_phases
        }
    return out


def render(cells: Dict[str, Dict[str, float]]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    if not cells:
        raise SystemExit("no cells parsed; did 10_tpcc_writeback_breakdown.sh run?")

    bar_specs = [(mode, f"tpccdeck W=128  {mode}") for mode in ("async", "sync")
                 if mode in cells]
    fig, ax = plt.subplots(figsize=(9.5, 2.4))
    y_positions = list(range(len(bar_specs)))[::-1]

    for y, (mode, label) in zip(y_positions, bar_specs):
        phases = cells[mode]
        left_us = 0.0
        for phase in PHASE_ORDER:
            v = phases.get(phase, 0)
            if v <= 0:
                continue
            v_ms = v / 1000.0
            ax.barh(
                y, v_ms, left=left_us / 1000.0,
                height=0.7,
                color=PHASE_COLORS[phase],
                edgecolor="black", linewidth=0.4,
            )
            left_us += v
        ax.text(
            left_us / 1000.0 + 0.5, y, f"{left_us / 1000.0:.1f} ms",
            va="center", ha="left", fontsize=12, color="#333",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in bar_specs], fontsize=13)
    ax.set_xlabel("Epoch wallclock (ms)")
    # ax.set_title("TPC-C tpccdeck W=128: per-phase epoch breakdown, async vs sync")
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    handles = [mpatches.Patch(color=PHASE_COLORS[p], label=PHASE_LABELS[p])
               for p in PHASE_ORDER]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.18),
        ncol=7, fontsize=12, frameon=False,
    )
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode"] + PHASE_ORDER + ["total_ms"])
        for mode in ("async", "sync"):
            if mode not in cells:
                continue
            phases = cells[mode]
            row = [mode] + [phases.get(p, 0) for p in PHASE_ORDER]
            row.append(round(sum(phases.get(p, 0) for p in PHASE_ORDER) / 1000.0, 2))
            w.writerow(row)
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(
            f"log dir not found: {LOG_DIR}; run 10_tpcc_writeback_breakdown.sh first"
        )
    cells = collect()
    render(cells)


if __name__ == "__main__":
    main()
