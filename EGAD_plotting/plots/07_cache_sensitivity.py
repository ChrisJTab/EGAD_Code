#!/usr/bin/env python3
"""Paper figure 07: hybrid_staging throughput vs cache/DB ratio.

Renders a single-panel cache-size sweep for ycsbf, 120 B, 20 M records, -rT-fF,
skew = 0.5. The X-axis is the cache/DB ratio (%); Y is steady-state throughput.

Three lines:
  hybrid_staging at varying cache caps    (the curve)
  gpu_only                                 (horizontal ceiling)
  cpu_only                                 (horizontal floor)

The "elbow" appears where the hybrid curve bends down toward cpu_only.

Reads $EGAD_LOGS_DIR/cache_sensitivity/*.log; writes
$EGAD_FIGURES_DIR/cache_sensitivity.{pdf,png,csv}.
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

LOG_DIR = os.path.join(LOGS_DIR, "cache_sensitivity")
OUT_BASE = os.path.join(FIGURES_DIR, "cache_sensitivity")

NTX_PER_EPOCH = 100_000
HYBRID_WINDOW = (250, 280)
BASELINE_WINDOW = (2, 5)   # gpu_only: 5-epoch run (no warmup needed on GPU-resident baseline)
CPU_WINDOW = (3, 10)       # cpu_only: average over epochs 3-10 (steady-state, post-warmup)

C_HYBRID = "#1f4e79"   # blue
C_GPU    = "#2e7d32"   # green ceiling
C_CPU    = "#b03a2e"   # red floor

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\].*Running epoch (\d+)"
)
HYBRID_FILE_RE = re.compile(
    r"ycsbf_hybrid_staging_cache(?P<pct>\d+)_rep(?P<rep>\d+)\.log$"
)
BASELINE_FILE_RE = re.compile(
    r"ycsbf_(?P<mode>gpu_only|cpu_only)_rep(?P<rep>\d+)\.log$"
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


def collect() -> Tuple[Dict[int, List[float]], Dict[str, List[float]]]:
    hybrid_by_pct: Dict[int, List[float]] = defaultdict(list)
    baseline_by_mode: Dict[str, List[float]] = defaultdict(list)

    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        name = os.path.basename(log)
        m = HYBRID_FILE_RE.search(name)
        if m:
            pct = int(m.group("pct"))
            mt = steady_state_mtxn(log, HYBRID_WINDOW)
            if mt is not None:
                hybrid_by_pct[pct].append(mt)
            continue
        m = BASELINE_FILE_RE.search(name)
        if m:
            mode = m.group("mode")
            window = CPU_WINDOW if mode == "cpu_only" else BASELINE_WINDOW
            mt = steady_state_mtxn(log, window)
            if mt is not None:
                baseline_by_mode[mode].append(mt)
            continue
    return hybrid_by_pct, baseline_by_mode


def render(
    hybrid_by_pct: Dict[int, List[float]],
    baseline_by_mode: Dict[str, List[float]],
) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    pcts = sorted(hybrid_by_pct.keys())
    if not pcts:
        raise SystemExit("no hybrid cells parsed; did 07_cache_sensitivity.sh run?")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ys = [statistics.mean(hybrid_by_pct[p]) for p in pcts]
    yerr = [[y - min(hybrid_by_pct[p]) for p, y in zip(pcts, ys)],
            [max(hybrid_by_pct[p]) - y for p, y in zip(pcts, ys)]]
    ax.errorbar(pcts, ys, yerr=yerr, capsize=2.5, elinewidth=1.0,
                color=C_HYBRID, linestyle="-", marker="o",
                markersize=9, linewidth=1.6, label="EGAD")

    if "gpu_only" in baseline_by_mode and baseline_by_mode["gpu_only"]:
        vs = baseline_by_mode["gpu_only"]
        y_gpu = statistics.mean(vs)
        ax.axhspan(min(vs), max(vs), color=C_GPU, alpha=0.12, linewidth=0)
        ax.axhline(y_gpu, color=C_GPU, linestyle="--", linewidth=1.2,
                   label=f"EPIC-GPU ({y_gpu:.1f})")
    if "cpu_only" in baseline_by_mode and baseline_by_mode["cpu_only"]:
        vs = baseline_by_mode["cpu_only"]
        y_cpu = statistics.mean(vs)
        ax.axhspan(min(vs), max(vs), color=C_CPU, alpha=0.12, linewidth=0)
        ax.axhline(y_cpu, color=C_CPU, linestyle="--", linewidth=1.2,
                   label=f"EPIC-CPU ({y_cpu:.1f})")

    ax.set_xlabel(r"cache / DB ratio (\%)")
    ax.set_ylabel("Throughput (MTxn/s)")
    # ax.set_title("YCSB-F, 120 B, 20 M, θ=0.5: throughput vs cache size")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, fontsize=13, frameon=False)
    ax.set_xticks([0, 5, 10, 25, 50, 75, 100])
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "cache_pct", "mean_mtxn_s", "min_mtxn_s", "max_mtxn_s", "n_reps"])
        for p in pcts:
            vs = hybrid_by_pct[p]
            w.writerow(["hybrid_staging", p, statistics.mean(vs),
                        round(min(vs), 3), round(max(vs), 3), len(vs)])
        for mode in ("gpu_only", "cpu_only"):
            vs = baseline_by_mode.get(mode, [])
            if vs:
                w.writerow([mode, "", statistics.mean(vs),
                            round(min(vs), 3), round(max(vs), 3), len(vs)])
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(
            f"log dir not found: {LOG_DIR}; run 07_cache_sensitivity.sh first"
        )
    hybrid_by_pct, baseline_by_mode = collect()
    render(hybrid_by_pct, baseline_by_mode)


if __name__ == "__main__":
    main()
