#!/usr/bin/env python3
"""Paper figure 16: epoch-size (batch-size) sensitivity, three systems.

Three panels:
  (a) YCSB-F throughput vs epoch size S -- EGAD keeps gaining past the
      canonical S=100K while EPIC-CPU flattens and EPIC-GPU peaks at
      100K and declines.
  (b) TPC-C (deck, W=64) throughput vs S -- same shapes.
  (c) YCSB-F throughput vs the latency each S implies (1.5x the epoch
      wallclock, Epic's convention): the latency-bound-throughput view.
      The same measurements as (a), re-plotted parametrically; rings
      mark S=100K on every curve and per-curve S labels show the
      direction of the sweep. EGAD is the only system that keeps
      converting latency into throughput through the swept range.

The gray vertical line in (a)/(b) marks S=100K, the operating point of
every other figure: it is the measured optimum of both baselines, so
the headline comparisons are conservative for EGAD.

Per-run metric: throughput and mean epoch wallclock over the arm's
steady-state window (the LAST window_txns/S epochs; window_txns is
5.28 M for EGAD YCSB, 5 M for stock YCSB, 4.8 M for TPC-C -- the runs
hold total transactions fixed so the window sits at identical table
state at every S). Epoch wallclock comes from the "Running epoch N"
log-timestamp deltas, the tail epoch from the last timestamped line.
Cells aggregate mean +/- std over reps.

Reads $EGAD_LOGS_DIR/epoch_size_sweep/*.log; writes
$EGAD_FIGURES_DIR/epoch_size_sweep.{pdf,png,csv}.
"""
from __future__ import annotations

import csv
import glob
import math
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
matplotlib.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["axes.formatter.use_mathtext"] = True  # avoid cmr10 missing-glyph warning
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

LOG_DIR = os.path.join(LOGS_DIR, "epoch_size_sweep")
OUT_BASE = os.path.join(FIGURES_DIR, "epoch_size_sweep")

# steady-state window, in transactions, per arm (see the test's header)
WINDOW_TXNS = {"egad_ycsb": 5_280_000, "stock": 5_000_000, "tpcc": 4_800_000}

FILENAME_RE = re.compile(
    r"^(?P<arm>egad|stockcpu|stockgpu|forkcpu|forkgpu)_(?P<wl>ycsbf|tpccdeck)"
    r"_s(?P<s>\d+)_rep(?P<rep>\d+)\.log$"
)
TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")
RUNNING_RE = re.compile(r"Running epoch (\d+)")

SERIES_STYLE = {
    "egad": dict(color="#1f4e79", linestyle="-", marker="o", label="EGAD"),
    "gpu":  dict(color="#2e7d32", linestyle="--", marker="s", label="EPIC-GPU"),
    "cpu":  dict(color="#b03a2e", linestyle="--", marker="^", label="EPIC-CPU"),
}
SERIES_ORDER = ["egad", "gpu", "cpu"]
ARM2SERIES = {"egad": "egad", "stockgpu": "gpu", "stockcpu": "cpu",
              "forkgpu": "gpu", "forkcpu": "cpu"}

CANONICAL_S = 100_000


def window_key(arm: str, wl: str) -> str:
    if arm == "egad" and wl == "ycsbf":
        return "egad_ycsb"
    if arm.startswith("stock"):
        return "stock"
    return "tpcc"


def parse_run(path: str, s: int, wtx: int) -> Optional[Tuple[float, float]]:
    """(throughput MTxn/s, mean epoch wallclock ms) over the steady window."""
    epoch_start: Dict[int, datetime] = {}
    last_ts: Optional[datetime] = None
    with open(path, errors="replace") as f:
        for line in f:
            m = TS_RE.match(line)
            if m:
                last_ts = datetime.strptime(m.group(1)[:26], "%Y-%m-%d %H:%M:%S.%f")
            m = RUNNING_RE.search(line)
            if m and last_ts is not None:
                epoch_start[int(m.group(1))] = last_ts
    es = sorted(epoch_start)
    if len(es) < 3:
        return None
    wall: Dict[int, float] = {}
    for a, b in zip(es, es[1:]):
        wall[a] = (epoch_start[b] - epoch_start[a]).total_seconds() * 1e3
    tail = (last_ts - epoch_start[es[-1]]).total_seconds() * 1e3
    if tail > 0:
        wall[es[-1]] = tail
    es = sorted(wall)
    nwin = max(3, math.ceil(wtx / s))
    if len(es) < nwin:
        print(f"    warn: {os.path.basename(path)} has {len(es)} epochs, "
              f"window needs {nwin}; skipped")
        return None
    win = es[-nwin:]
    wall_ms = [wall[e] for e in win]
    tput = (nwin * s) / (sum(wall_ms) / 1e3) / 1e6
    return tput, statistics.mean(wall_ms)


def collect() -> Dict[Tuple[str, str], Dict[int, List[Tuple[float, float]]]]:
    cells: Dict[Tuple[str, str], Dict[int, List[Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        m = FILENAME_RE.match(os.path.basename(log))
        if not m:
            continue
        arm, wl, s = m.group("arm"), m.group("wl"), int(m.group("s"))
        r = parse_run(log, s, WINDOW_TXNS[window_key(arm, wl)])
        if r is not None:
            cells[(wl, ARM2SERIES[arm])][s].append(r)
    return cells


def kfmt(v: float) -> str:
    return f"{int(v / 1000)}K" if v >= 1000 else str(int(v))


def render(cells) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)
    if not cells:
        raise SystemExit(f"no cells parsed under {LOG_DIR}; "
                         "did 16_epoch_size_sweep.sh run?")

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 3.0))
    tick_s = [5000, 25000, 100000, 400000]

    # (a)/(b): throughput vs epoch size
    for ax, wl, title in (
        (axes[0], "ycsbf", r"YCSB-F ($\theta = 0.5$, 20 M $\times$ 120 B)"),
        (axes[1], "tpccdeck", r"TPC-C ($W = 64$)"),
    ):
        for series in SERIES_ORDER:
            data = cells.get((wl, series))
            if not data:
                continue
            ss = sorted(data)
            y = [statistics.mean(t for t, _ in data[s]) for s in ss]
            ye = [statistics.stdev([t for t, _ in data[s]]) if len(data[s]) > 1 else 0.0
                  for s in ss]
            ax.errorbar(ss, y, yerr=ye, capsize=2.5, linewidth=1.6, markersize=7,
                        **SERIES_STYLE[series])
        ax.set_xscale("log")
        ax.set_xticks(tick_s)
        ax.set_xticklabels([kfmt(s) for s in tick_s])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.axvline(CANONICAL_S, color="0.75", linewidth=0.9, zorder=0)
        ax.set_xlabel("epoch size (txns)", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(axis="both", labelsize=12)
        ax.set_ylim(bottom=0)

    # (c): throughput vs implied latency, YCSB-F. Same measurements as (a);
    # x is the latency each epoch size implies for that system, so the curves
    # are parametric in S and do not share x positions (faster systems sit
    # left). Rings mark the canonical S=100K point on every curve, and epoch-
    # size labels on each curve make the parametric direction visible.
    ax = axes[2]
    LABEL_SPECS = {  # series -> {S: (dx_pt, dy_pt, ha)}
        "egad": {5000: (6, -11, "left"), 100000: (10, -15, "left"),
                 400000: (6, -11, "left")},
        "cpu":  {5000: (6, 5, "left"), 400000: (-2, 8, "center")},
        "gpu":  {5000: (0, 8, "center"), 100000: (0, -17, "center"),
                 400000: (0, 8, "center")},
    }
    for series in SERIES_ORDER:
        data = cells.get(("ycsbf", series))
        if not data:
            continue
        ss = sorted(data)
        x = [1.5 * statistics.mean(w for _, w in data[s]) for s in ss]
        y = [statistics.mean(t for t, _ in data[s]) for s in ss]
        ax.plot(x, y, linewidth=1.6, markersize=7, **SERIES_STYLE[series])
        col = SERIES_STYLE[series]["color"]
        if CANONICAL_S in ss:
            i = ss.index(CANONICAL_S)
            ax.plot(x[i], y[i], marker="o", markersize=14,
                    markerfacecolor="none", markeredgecolor=col,
                    markeredgewidth=1.4, linestyle="none", zorder=5)
        for s, (dx, dy, ha) in LABEL_SPECS.get(series, {}).items():
            if s not in ss:
                continue
            i = ss.index(s)
            ax.annotate(kfmt(s), (x[i], y[i]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=9, ha=ha, color=col)
    ax.set_xscale("log")
    ax.set_xlim(0.28, 95)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"avg latency, $1.5\times$ epoch (ms)", fontsize=13)
    ax.set_title("YCSB-F, latency-bound", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Throughput (MTxn/s)", fontsize=13)
    handles = [plt.Line2D([], [], markersize=7, linewidth=1.6, **SERIES_STYLE[k])
               for k in SERIES_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
               ncol=3, fontsize=13, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    with open(OUT_BASE + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "series", "epoch_size", "tput_mean_mtxns",
                    "tput_std_mtxns", "latency_ms_15x", "n_reps"])
        for wl in ("ycsbf", "tpccdeck"):
            for series in SERIES_ORDER:
                data = cells.get((wl, series), {})
                for s in sorted(data):
                    ts = [t for t, _ in data[s]]
                    ws = [w_ for _, w_ in data[s]]
                    w.writerow([wl, series, s,
                                round(statistics.mean(ts), 3),
                                round(statistics.stdev(ts), 3) if len(ts) > 1 else 0.0,
                                round(1.5 * statistics.mean(ws), 2),
                                len(ts)])
    print(f"saved {OUT_BASE}.csv")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(
            f"log dir not found: {LOG_DIR}; run 16_epoch_size_sweep.sh first"
        )
    render(collect())


if __name__ == "__main__":
    main()
