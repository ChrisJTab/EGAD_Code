#!/usr/bin/env python3
"""Paper figure 11: TPCC deep-E cliff demonstration.

X-axis: epoch number (1 to 400).
Y-axis: per-epoch throughput in MTxn/s (instantaneous, NOT windowed).

Two solid lines + one horizontal floor:
  - tpccdeck (blue):  flat at ~6.5 MTxn/s through e=400 (no cliff because
                      NewOrder rate == Delivery rate keeps OL backlog
                      bounded at ~9000W = 1.15M rows).
  - tpccfull (red):   climbs early, then cliffs from ~e133 onward as the
                      unbounded undelivered queue outgrows the cache and
                      eviction starts hitting still-needed slots.
  - cpu_only (gray):  horizontal dashed line at the cpu_only e30-50
                      median (the post-cliff floor that tpccfull
                      asymptotes toward).

Per-epoch throughput is `100000 / (t_epoch_n+1 - t_epoch_n)`, taken from
the `Running epoch N` timestamps. Median across 3 reps per (mix, epoch).

Reads $EGAD_LOGS_DIR/tpcc_cliff/*.log; writes
$EGAD_FIGURES_DIR/tpcc_cliff.{pdf,png,csv}.
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

LOG_DIR = os.path.join(LOGS_DIR, "tpcc_cliff")
OUT_BASE = os.path.join(FIGURES_DIR, "tpcc_cliff")

NTX_PER_EPOCH = 100_000
CPU_WINDOW = (30, 50)

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\].*Running epoch (\d+)"
)
HYBRID_FILE_RE = re.compile(
    r"(?P<mix>tpccdeck|tpccfull)_w128_hybrid_reserve6_rep(?P<rep>\d+)\.log$"
)
CPU_FILE_RE = re.compile(r"tpccdeck_w128_cpu_only_floor_rep(?P<rep>\d+)\.log$")

MIX_STYLE = {
    "tpccdeck": dict(color="#1f4e79", linestyle="-", marker=None,
                     linewidth=1.6, label="tpccdeck (bounded)"),
    "tpccfull": dict(color="#b03a2e", linestyle="-", marker=None,
                     linewidth=1.6, label="tpccfull (cliffs)"),
}
CPU_STYLE = dict(color="#666666", linestyle="--", linewidth=1.2,
                 label="EPIC-CPU floor")


def parse_per_epoch_mtxn(log_path: str) -> Dict[int, float]:
    """Return {epoch: instantaneous_mtxn_per_s} from log_path."""
    times: Dict[int, datetime] = {}
    with open(log_path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                ts = datetime.strptime(
                    f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S.%f"
                )
                times[int(m.group(3))] = ts

    eps = sorted(times.keys())
    out: Dict[int, float] = {}
    for i in range(len(eps) - 1):
        dt = (times[eps[i + 1]] - times[eps[i]]).total_seconds()
        if dt > 0:
            out[eps[i]] = NTX_PER_EPOCH / dt / 1e6
    return out


def windowed_mtxn(log_path: str, window: Tuple[int, int]) -> Optional[float]:
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


def collect() -> Tuple[Dict[str, Dict[int, List[float]]], Optional[float]]:
    hybrid: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    cpu_vals: List[float] = []
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        name = os.path.basename(log)
        m = HYBRID_FILE_RE.search(name)
        if m:
            mix = m.group("mix")
            for ep, mt in parse_per_epoch_mtxn(log).items():
                hybrid[mix][ep].append(mt)
            continue
        m = CPU_FILE_RE.search(name)
        if m:
            mt = windowed_mtxn(log, CPU_WINDOW)
            if mt is not None:
                cpu_vals.append(mt)
    cpu_floor = statistics.median(cpu_vals) if cpu_vals else None
    return hybrid, cpu_floor


def render(hybrid: Dict[str, Dict[int, List[float]]], cpu_floor: Optional[float]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    if not hybrid:
        raise SystemExit("no cliff cells parsed; did 11_tpcc_cliff.sh run?")

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for mix in ("tpccdeck", "tpccfull"):
        if mix not in hybrid:
            continue
        eps = sorted(hybrid[mix].keys())
        ys = [statistics.median(hybrid[mix][e]) for e in eps]
        ax.plot(eps, ys, **MIX_STYLE[mix])

    if cpu_floor is not None:
        ax.axhline(cpu_floor, **CPU_STYLE)

    ax.set_xlabel("epoch")
    ax.set_ylabel("Throughput (MTxn/s)")
    # ax.set_title("TPC-C W=128, --hybrid_hbm_reserve_gb=6: deep-E cliff under tight cache")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=13, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mix", "epoch", "median_mtxn_s", "n_reps"])
        for mix in ("tpccdeck", "tpccfull"):
            if mix not in hybrid:
                continue
            for ep in sorted(hybrid[mix].keys()):
                vs = hybrid[mix][ep]
                w.writerow([mix, ep, round(statistics.median(vs), 3), len(vs)])
        if cpu_floor is not None:
            w.writerow(["cpu_only_floor", "", round(cpu_floor, 3), ""])
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(
            f"log dir not found: {LOG_DIR}; run 11_tpcc_cliff.sh first"
        )
    hybrid, cpu_floor = collect()
    render(hybrid, cpu_floor)


if __name__ == "__main__":
    main()
