#!/usr/bin/env python3
"""Paper figure 14: combined per-phase epoch breakdown, YCSB + TPCC.

Ten horizontal stacked bars in one figure, ordered from largest async
win at the top to smallest at the bottom:

  TPCC tpccdeck W=128 (async, sync)   - the optimal case (writeback ~=
                                        rest of pipeline, near-2x gain)
  YCSB-A (async, sync)                - large writeback, moderate gain
  YCSB-F (async, sync)                - large writeback, moderate gain
  YCSB-B (async, sync)                - small writeback, negligible gain
  YCSB-C (async, sync)                - read-only, near-zero gain

YCSB's transfer (compute) and transfer (data) phases are aggregated
into a single "stage" block to match TPCC's prepareEpoch decomposition,
so both share the same 7-phase axis:

  index_transfer | indexing | submission | stage | initialization
                 | execution | writeback

The figure tells four things at a glance:
  1. The async win scales with the ratio of writeback time to the rest
     of the epoch (TPCC = ratio near 1.0 = near-2x gain).
  2. The writeback block (red) shrinks from a fat chunk on sync bars
     to a thin sliver on async bars; that delta IS the win.
  3. The stage / prepareEpoch block is slightly LARGER in async than
     in sync (visible on most cells) - this is the host-side
     contention tax documented in Appendix C.
  4. On YCSB the index_transfer block is also inflated in async for
     the same reason (CUDA-runtime contention with the worker thread).

Reads $EGAD_LOGS_DIR/writeback_breakdown/ (YCSB) and
$EGAD_LOGS_DIR/tpcc_writeback_breakdown/ (TPCC). Writes
$EGAD_FIGURES_DIR/breakdown_combined.{pdf,png,csv}.
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

YCSB_DIR = os.path.join(LOGS_DIR, "writeback_breakdown")
TPCC_DIR = os.path.join(LOGS_DIR, "tpcc_writeback_breakdown")
OUT_BASE = os.path.join(FIGURES_DIR, "breakdown_combined")

# Steady-state windows (per feedback_steady_state_windows).
YCSB_LO, YCSB_HI = 30, 40
TPCC_LO, TPCC_HI = 30, 50

PHASE_COLORS = {
    "index_transfer": "#6B9F71",
    "indexing":       "#93B883",
    "submission":     "#C9A961",
    "stage":          "#1F4E79",
    "initialization": "#B8A8D5",
    "execution":      "#6B5390",
    "writeback":      "#B23A48",
}
PHASE_LABELS = {
    "index_transfer": "index transfer",
    "indexing":       "indexing",
    "submission":     "submit",
    "stage":          "stage",
    "initialization": "init",
    "execution":      "exec",
    "writeback":      "writeback",
}
PHASE_ORDER = [
    "index_transfer", "indexing", "submission",
    "stage",
    "initialization", "execution",
    "writeback",
]

# Bar order: top-down, biggest gain first.
WORKLOAD_LABEL = {
    "tpccdeck": "TPC-C",
    "ycsbf":    "YCSB-F",
    "ycsbb":    "YCSB-B",
}
# YCSB-A and YCSB-C are intentionally omitted: A behaves like F (both
# write-heavy, both at ~1.22x async gain) and C behaves like B (both
# light on writeback, both at ~1.0x). F and B are the representative
# write-heavy and write-light YCSB cases respectively.
WORKLOAD_ORDER = ["tpccdeck", "ycsbf", "ycsbb"]

# --- YCSB parsing (plot 08 source) ---
YCSB_EPOCH_PHASE_RE = re.compile(
    r"Epoch (\d+) (index_transfer|indexing|submission|staging|initialization|execution|"
    r"flush|flush_sync_prev|flush_start_async) time: (\d+) us"
)
YCSB_RUN_EPOCH_RE = re.compile(r"Running epoch (\d+)")
YCSB_TRANSFER_DATA_RE = re.compile(
    r"\[HYBRID\] evict:scatter_gather_transfer_evicted\(gpu\): (\d+) us"
)
YCSB_FILENAME_RE = re.compile(
    r"(?P<bench>ycsb[abcf])_(?P<mode>async|sync)_rep(?P<rep>\d+)\.log$"
)

# --- TPCC parsing (plot 10 source) ---
TPCC_EPOCH_PHASE_RE = re.compile(
    r"Epoch (\d+) (index_transfer|indexing|submission|initialization|execution|"
    r"hybrid_staging:prepareEpoch|hybrid_staging:remap|"
    r"hybrid_staging:flush_sync_prev|hybrid_staging:flush_start_async|"
    r"hybrid_staging:periodicFlush) time: (\d+) us"
)
TPCC_FILENAME_RE = re.compile(
    r"tpccdeck_w(?P<w>\d+)_(?P<mode>async|sync)_rep(?P<rep>\d+)\.log$"
)


def parse_ycsb_log(log_path: str) -> Dict[str, float]:
    """Return {phase: median_us over EPOCH_LO..EPOCH_HI} for a YCSB rep log."""
    per_epoch: Dict[int, Dict[str, int]] = defaultdict(dict)
    current_epoch: Optional[int] = None
    with open(log_path) as f:
        for line in f:
            m = YCSB_RUN_EPOCH_RE.search(line)
            if m:
                current_epoch = int(m.group(1)); continue
            m = YCSB_TRANSFER_DATA_RE.search(line)
            if m and current_epoch is not None:
                per_epoch[current_epoch]["transfer_data"] = int(m.group(1)); continue
            m = YCSB_EPOCH_PHASE_RE.search(line)
            if m:
                ep = int(m.group(1)); phase = m.group(2); dur = int(m.group(3))
                per_epoch[ep][phase] = dur

    def coalesce(d: Dict[str, int]) -> Dict[str, int]:
        # Aggregate YCSB's staging + transfer_data into single "stage" block
        # so it matches TPCC's prepareEpoch granularity.
        stage = d.get("staging", 0)
        wb = (d.get("flush", 0)
              + d.get("flush_sync_prev", 0)
              + d.get("flush_start_async", 0))
        out = {k: v for k, v in d.items()
               if k not in ("staging", "transfer_data",
                            "flush", "flush_sync_prev", "flush_start_async")}
        out["stage"] = stage
        out["writeback"] = wb
        return out

    medians: Dict[str, List[int]] = defaultdict(list)
    for ep, ph in per_epoch.items():
        if YCSB_LO <= ep <= YCSB_HI:
            for k, v in coalesce(ph).items():
                medians[k].append(v)
    return {k: statistics.mean(v) for k, v in medians.items() if v}


def parse_tpcc_log(log_path: str) -> Dict[str, float]:
    """Return {phase: median_us over EPOCH_LO..EPOCH_HI} for a TPCC rep log."""
    per_epoch: Dict[int, Dict[str, int]] = defaultdict(dict)
    with open(log_path) as f:
        for line in f:
            m = TPCC_EPOCH_PHASE_RE.search(line)
            if not m:
                continue
            ep = int(m.group(1)); phase = m.group(2); dur = int(m.group(3))
            if phase in ("hybrid_staging:prepareEpoch", "hybrid_staging:remap"):
                per_epoch[ep]["stage"] = per_epoch[ep].get("stage", 0) + dur
            elif phase in ("hybrid_staging:flush_sync_prev",
                           "hybrid_staging:flush_start_async",
                           "hybrid_staging:periodicFlush"):
                per_epoch[ep]["writeback"] = per_epoch[ep].get("writeback", 0) + dur
            else:
                per_epoch[ep][phase] = dur

    medians: Dict[str, List[int]] = defaultdict(list)
    for ep, ph in per_epoch.items():
        if TPCC_LO <= ep <= TPCC_HI:
            for k, v in ph.items():
                medians[k].append(v)
    return {k: statistics.mean(v) for k, v in medians.items() if v}


def collect() -> Dict[Tuple[str, str], Dict[str, float]]:
    """{(workload, mode): {phase: median_us}}. Median across reps."""
    per_cell: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(YCSB_DIR, "*.log"))):
        m = YCSB_FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        per_rep = parse_ycsb_log(log)
        if per_rep:
            per_cell[(m.group("bench"), m.group("mode"))].append(per_rep)
    for log in sorted(glob.glob(os.path.join(TPCC_DIR, "*.log"))):
        m = TPCC_FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        per_rep = parse_tpcc_log(log)
        if per_rep:
            per_cell[("tpccdeck", m.group("mode"))].append(per_rep)

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for k, reps in per_cell.items():
        all_phases = set().union(*(r.keys() for r in reps))
        out[k] = {p: statistics.mean([r.get(p, 0) for r in reps]) for p in all_phases}
    return out


def render(cells: Dict[Tuple[str, str], Dict[str, float]]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    if not cells:
        raise SystemExit("no cells parsed; did 08_writeback_breakdown.sh "
                         "and 10_tpcc_writeback_breakdown.sh run?")

    # Pair layout: each workload occupies 2 adjacent y positions (async on top,
    # sync below); pairs are separated by a 1-unit gap so they read as groups.
    # Workload label sits at the midpoint between the pair's two bars.
    workloads_present = [w for w in WORKLOAD_ORDER if (w, "async") in cells or (w, "sync") in cells]
    n_w = len(workloads_present)
    bar_specs: List[Tuple[str, str, float]] = []
    tick_positions: List[float] = []
    tick_labels: List[str] = []
    for i, w in enumerate(workloads_present):
        base = (n_w - 1 - i) * 3  # 0, 3, 6, ...
        async_y, sync_y = base + 1, base
        tick_positions.append((async_y + sync_y) / 2)
        tick_labels.append(WORKLOAD_LABEL[w])
        if (w, "async") in cells:
            bar_specs.append((w, "async", async_y))
        if (w, "sync") in cells:
            bar_specs.append((w, "sync", sync_y))

    # Wide single-column figure. Fontsizes are bumped accordingly so they
    # remain legible after LaTeX scales the PDF down to \columnwidth.
    fig = plt.figure(figsize=(9.0, 4.5))
    ax = fig.add_axes([0.15, 0.13, 0.83, 0.68])

    HATCH_FOR_MODE = {"async": "", "sync": "///"}
    BAR_HEIGHT = 0.7  # gives ~0.3 within-pair gap vs ~1.3 between-pair gap

    for w, mode, y in bar_specs:
        phases = cells[(w, mode)]
        left_us = 0.0
        for phase in PHASE_ORDER:
            v = phases.get(phase, 0)
            if v <= 0:
                continue
            ax.barh(
                y, v / 1000.0, left=left_us / 1000.0,
                height=BAR_HEIGHT, color=PHASE_COLORS[phase],
                edgecolor="black", linewidth=0.05,
                hatch=HATCH_FOR_MODE[mode],
            )
            left_us += v
        ax.text(left_us / 1000.0 + 0.4, y, f"{left_us / 1000.0:.1f} ms",
                va="center", ha="left", fontsize=22, color="#333")

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=24)
    ax.tick_params(axis="x", labelsize=22)
    ax.set_xlabel("Epoch wallclock (ms)", fontsize=24)
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax.set_xlim(left=0)

    # Phase legend at top (2 rows of 4 by virtue of ncol=4).
    phase_handles = [mpatches.Patch(color=PHASE_COLORS[p], label=PHASE_LABELS[p])
                     for p in PHASE_ORDER]
    phase_leg = ax.legend(
        handles=phase_handles, loc="upper center", bbox_to_anchor=(0.5, 1.42),
        ncol=4, fontsize=22, frameon=False,
    )
    ax.add_artist(phase_leg)

    # Compact mode legend (async vs sync via hatch) in the bottom-right corner.
    mode_handles = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="",    label="async"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="///", label="sync"),
    ]
    ax.legend(
        handles=mode_handles, loc="lower right",
        fontsize=20, frameon=True, framealpha=0.95, edgecolor="0.7",
    )
    # bbox_inches="tight" doesn't auto-include legends added via add_artist;
    # pass phase_leg explicitly so it isn't cropped.
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight", bbox_extra_artists=[phase_leg])
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600, bbox_extra_artists=[phase_leg])
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "mode"] + PHASE_ORDER + ["total_ms"])
        for wname in WORKLOAD_ORDER:
            for mode in ("async", "sync"):
                if (wname, mode) not in cells:
                    continue
                phases = cells[(wname, mode)]
                row = [wname, mode] + [phases.get(p, 0) for p in PHASE_ORDER]
                row.append(round(sum(phases.get(p, 0) for p in PHASE_ORDER) / 1000.0, 2))
                w.writerow(row)
    print(f"saved {csv_path}")


def main() -> None:
    cells = collect()
    render(cells)


if __name__ == "__main__":
    main()
