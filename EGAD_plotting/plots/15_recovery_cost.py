#!/usr/bin/env python3
"""Paper figure 15: the cost of GPU-crash recovery on TPC-C.

Standalone "how expensive is recovery" figure. For each warehouse count it
shows two hybrid_staging bars on the SAME build (epic@e66f6af, the durable
recovery path):

  off = non-recovery hybrid_staging (canonical fig-09 config)
  on  = provisioned recovery (off + durable Primary Store + prefault)

so the off->on gap is pure recovery overhead, annotated as a percentage above
each pair. A dashed reference line marks the stock EPIC cpu_only floor, making
the point that recovery-on still beats the CPU fallback at every W >= 8.

tpccdeck mix, e=50, steady-state window e30-50. Reads
$EGAD_LOGS_DIR/recovery_cost/*.log; writes
$EGAD_FIGURES_DIR/recovery_cost.{pdf,png,csv}.
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
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["axes.formatter.use_mathtext"] = True  # cmr10 + mathtext tick labels
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

LOG_DIR = os.path.join(LOGS_DIR, "recovery_cost")
OUT_BASE = os.path.join(FIGURES_DIR, "recovery_cost")

NTX_PER_EPOCH = 100_000
WINDOW = (30, 50)  # TPC-C steady-state (feedback_steady_state_windows)

C_OFF = "#1f4e79"  # non-recovery hybrid (blue)
C_ON = "#b03a2e"   # recovery hybrid (red)
C_CPU = "#555555"  # cpu_only floor (gray dashed)

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\].*Running epoch (\d+)"
)
FILENAME_RE = re.compile(
    r"tpccdeck_(?P<mode>off|on|cpu)_w(?P<w>\d+)_rep(?P<rep>\d+)\.log$"
)


def steady_state_mtxn(log_path: str, window: Tuple[int, int]) -> Optional[float]:
    times: Dict[int, datetime] = {}
    with open(log_path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                times[int(m.group(3))] = datetime.strptime(
                    f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S.%f"
                )
    a, b = window
    if a not in times or b not in times:
        return None
    dt = (times[b] - times[a]).total_seconds()
    if dt <= 0:
        return None
    return ((b - a) * NTX_PER_EPOCH) / dt / 1e6


def collect() -> Dict[Tuple[str, int], List[float]]:
    by_cell: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        m = FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        mt = steady_state_mtxn(log, WINDOW)
        if mt is not None:
            by_cell[(m.group("mode"), int(m.group("w")))].append(mt)
    return by_cell


def render(cells: Dict[Tuple[str, int], List[float]]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    ws = sorted({w for (_, w) in cells.keys()})
    if not ws:
        raise SystemExit("no cells parsed; did 15_recovery_cost.sh run?")

    def med(mode: str, w: int) -> Optional[float]:
        vs = cells.get((mode, w), [])
        return statistics.median(vs) if vs else None

    off = [med("off", w) for w in ws]
    on = [med("on", w) for w in ws]
    cpu = [med("cpu", w) for w in ws]

    x = list(range(len(ws)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    b_off = ax.bar([xi - width / 2 for xi in x], off, width,
                   color=C_OFF, label="EGAD (no recovery)")
    b_on = ax.bar([xi + width / 2 for xi in x], on, width,
                  color=C_ON, label="EGAD (with recovery)")

    # cpu_only floor as a dashed step reference across the W groups.
    if all(c is not None for c in cpu):
        ax.plot(x, cpu, color=C_CPU, linestyle="--", linewidth=1.4,
                marker="x", markersize=6, label="EPIC-CPU")

    # Annotate recovery overhead % above each off/on pair.
    for xi, (o, n) in enumerate(zip(off, on)):
        if o and n:
            pct = (o - n) / o * 100.0
            top = max(o, n)
            ax.annotate(f"$-{pct:.0f}\\%$", xy=(xi, top), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=11, color=C_ON)

    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in ws])
    ax.set_xlabel("Warehouse count $W$", fontsize=13)
    ax.set_ylabel("Throughput (MTxn/s)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    ymax = max([v for v in off + on + cpu if v is not None]) * 1.12
    ax.set_ylim(0, ymax)
    # Legend above the axes so it never collides with the bars or the
    # overhead annotations (the bars span the full width).
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), fontsize=11,
              frameon=False, ncol=3, columnspacing=1.6, handletextpad=0.5)

    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["W", "off_mtxn_s", "on_mtxn_s", "cpu_mtxn_s",
                    "recovery_overhead_pct", "on_x_cpu", "n_reps_on"])
        for i, wh in enumerate(ws):
            o, n, c = off[i], on[i], cpu[i]
            pct = (o - n) / o * 100.0 if (o and n) else ""
            ratio = n / c if (n and c) else ""
            w.writerow([wh, o, n, c, pct, ratio, len(cells.get(("on", wh), []))])
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(f"log dir not found: {LOG_DIR}; run 15_recovery_cost.sh first")
    render(collect())


if __name__ == "__main__":
    main()
