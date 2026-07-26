#!/usr/bin/env python3
"""Paper figure 08: per-phase epoch breakdown, async vs sync writeback.

Renders 8 horizontal stacked bars (4 workloads x {async, sync}). Each bar is
the steady-state epoch wallclock, decomposed into 7 colored phase blocks:

  index_transfer, indexing, submission, staging, initialization, execution,
  writeback

where "writeback" = flush in sync mode (the whole synchronous writeback)
and flush_sync_prev + flush_start_async in async mode (only the critical-path
slivers; the real D2H/scatter work runs concurrently on a worker thread and is
invisible to the main-thread wallclock).

The visual punchline: in sync mode the writeback block is a big colored chunk
on the right of each bar; in async mode it shrinks to a thin sliver and the
total bar is shorter (= higher throughput).

Per-phase values are the median across epochs e30-e40 of each rep, then median
across reps.

Reads $EGAD_LOGS_DIR/writeback_breakdown/*.log; writes
$EGAD_FIGURES_DIR/writeback_breakdown.{pdf,png,csv}.
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

LOG_DIR = os.path.join(LOGS_DIR, "writeback_breakdown")
OUT_BASE = os.path.join(FIGURES_DIR, "writeback_breakdown")

# Steady-state per-epoch window (40-epoch run; e30-e40 is past warmup).
EPOCH_LO, EPOCH_HI = 30, 40

WORKLOAD_TITLES = {
    "ycsba": "YCSB-A",
    "ycsbb": "YCSB-B",
    "ycsbc": "YCSB-C",
    "ycsbf": "YCSB-F",
}
WORKLOAD_ORDER = ["ycsba", "ycsbb", "ycsbc", "ycsbf"]

# Phase color scheme: indexing greens, submission mustard, staging blue,
# init/exec purples, writeback red (so the sync flush block stands out).
PHASE_COLORS = {
    "index_transfer": "#6B9F71",   # green
    "indexing":       "#93B883",   # lighter green
    "submission":     "#C9A961",   # mustard
    "transfer_compute": "#7AA3CC",   # light blue (working-set compute + remap)
    "transfer_data":    "#1F4E79",   # deep blue (PCIe + scatter)
    "initialization": "#B8A8D5",   # lavender
    "execution":      "#6B5390",   # deep purple
    "writeback":      "#B23A48",   # red
}
PHASE_LABELS = {
    "index_transfer": "idx tsf",
    "indexing":       "indexing",
    "submission":     "submit",
    "transfer_compute": "transfer (compute)",
    "transfer_data":    "transfer (data)",
    "initialization": "init",
    "execution":      "exec",
    "writeback":      "writeback",
}
PHASE_ORDER = [
    "index_transfer", "indexing", "submission",
    "transfer_compute", "transfer_data",
    "initialization", "execution",
    "writeback",
]

EPOCH_PHASE_RE = re.compile(
    r"Epoch (\d+) (index_transfer|indexing|submission|staging|initialization|execution|"
    r"flush|flush_sync_prev|flush_start_async) time: (\d+) us"
)
RUN_EPOCH_RE = re.compile(r"Running epoch (\d+)")
TRANSFER_DATA_RE = re.compile(
    r"\[HYBRID\] evict:scatter_gather_transfer_evicted\(gpu\): (\d+) us"
)
FILENAME_RE = re.compile(
    r"(?P<bench>ycsb[abcf])_(?P<mode>async|sync)_rep(?P<rep>\d+)\.log$"
)


def parse_per_epoch(log_path: str) -> Dict[str, float]:
    """Return median-per-phase across EPOCH_LO..EPOCH_HI for this single log."""
    per_epoch: Dict[int, Dict[str, int]] = defaultdict(dict)
    current_epoch: Optional[int] = None
    with open(log_path) as f:
        for line in f:
            m = RUN_EPOCH_RE.search(line)
            if m:
                current_epoch = int(m.group(1))
                continue
            m = TRANSFER_DATA_RE.search(line)
            if m and current_epoch is not None:
                per_epoch[current_epoch]["transfer_data"] = int(m.group(1))
                continue
            m = EPOCH_PHASE_RE.search(line)
            if m:
                ep = int(m.group(1))
                phase = m.group(2)
                dur = int(m.group(3))
                per_epoch[ep][phase] = dur

    def coalesce(epoch_phases: Dict[str, int]) -> Dict[str, int]:
        # Split staging into transfer_data + transfer_compute.
        staging = epoch_phases.get("staging", 0)
        sg = epoch_phases.get("transfer_data", 0)
        transfer_compute = max(0, staging - sg)
        # Collapse writeback markers.
        wb = (
            epoch_phases.get("flush", 0)
            + epoch_phases.get("flush_sync_prev", 0)
            + epoch_phases.get("flush_start_async", 0)
        )
        out = {k: v for k, v in epoch_phases.items()
               if k not in ("flush", "flush_sync_prev", "flush_start_async",
                            "staging", "transfer_data")}
        out["transfer_compute"] = transfer_compute
        out["transfer_data"] = sg
        out["writeback"] = wb
        return out

    medians: Dict[str, List[int]] = defaultdict(list)
    for ep, ph in per_epoch.items():
        if EPOCH_LO <= ep <= EPOCH_HI:
            for k, v in coalesce(ph).items():
                medians[k].append(v)

    return {k: statistics.median(v) for k, v in medians.items() if v}


def collect() -> Dict[Tuple[str, str], Dict[str, float]]:
    """Return {(workload, mode): {phase: median_us}}."""
    per_cell: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        m = FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        bench = m.group("bench")
        mode = m.group("mode")
        per_rep = parse_per_epoch(log)
        if per_rep:
            per_cell[(bench, mode)].append(per_rep)

    # Median across reps per phase.
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (bench, mode), reps in per_cell.items():
        all_phases = set().union(*(r.keys() for r in reps))
        out[(bench, mode)] = {
            phase: statistics.median([r.get(phase, 0) for r in reps])
            for phase in all_phases
        }
    return out


def render(cells: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    if not cells:
        raise SystemExit("no cells parsed; did 08_writeback_breakdown.sh run?")

    # Bar order: per workload, async then sync (so async-vs-sync are adjacent).
    bar_specs: List[Tuple[str, str, str]] = []  # (bench, mode, label)
    for bench in WORKLOAD_ORDER:
        for mode in ("async", "sync"):
            if (bench, mode) in cells:
                label = f"{WORKLOAD_TITLES[bench]}  {mode}"
                bar_specs.append((bench, mode, label))

    fig_height = 0.5 * len(bar_specs) + 1.5
    fig, ax = plt.subplots(figsize=(9.5, fig_height))

    y_positions = list(range(len(bar_specs)))[::-1]   # top to bottom

    for y, (bench, mode, label) in zip(y_positions, bar_specs):
        phases = cells[(bench, mode)]
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
        # Total label on the right of the bar.
        total_ms = left_us / 1000.0
        ax.text(
            total_ms + 0.15, y, f"{total_ms:.1f} ms",
            va="center", ha="left", fontsize=12, color="#333",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, _, label in bar_specs], fontsize=13)
    ax.set_xlabel("Epoch wallclock (ms)")
    # ax.set_title(
    #     "YCSB -rT-fF, 120 B, 20 M, θ=0.5, 33.6% cache: per-phase epoch breakdown"
    # )
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")

    # Legend.
    handles = [mpatches.Patch(color=PHASE_COLORS[p], label=PHASE_LABELS[p])
               for p in PHASE_ORDER]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
        ncol=7, fontsize=12, frameon=False,
    )

    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    # CSV
    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "mode"] + PHASE_ORDER + ["total_ms"])
        for bench in WORKLOAD_ORDER:
            for mode in ("async", "sync"):
                if (bench, mode) not in cells:
                    continue
                phases = cells[(bench, mode)]
                row = [bench, mode] + [phases.get(p, 0) for p in PHASE_ORDER]
                row.append(sum(phases.get(p, 0) for p in PHASE_ORDER) / 1000.0)
                w.writerow(row)
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(
            f"log dir not found: {LOG_DIR}; run 08_writeback_breakdown.sh first"
        )
    cells = collect()
    render(cells)


if __name__ == "__main__":
    main()
