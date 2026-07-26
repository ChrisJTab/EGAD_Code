#!/usr/bin/env python3
"""Paper figure 12: YCSB cache-warmup trajectory at 1 KB records, all 6 skews.

Reads $EGAD_LOGS_DIR/record_size/1KB_hybrid_staging_skew*.log (shared
with plot 03). Renders one line per skew using the per-epoch
throughput across a 300-epoch ycsbf hybrid_staging run at 1 KB
records, 20 M records, autosizer's natural ~52.7 % cache/DB ratio.

Each line starts low at epoch 1 (cold cache, every read is a miss-
admit, big admit volume since 1 KB records ship 8x more bytes than
120 B). As the working set settles in cache, throughput climbs until
admit volume crosses cache capacity and FIFO eviction begins, at
which point the curve plateaus at its post-eviction steady state.

A thin red dashed vertical line per skew marks **the eviction-onset
epoch** (the first epoch the stager logs `Evictions needed:`,
i.e. when the per-epoch admit set first exceeds free-list slots and
the FIFO eviction path triggers). Skews are positioned at their
median onset epoch across 3 reps.

Writes $EGAD_FIGURES_DIR/ycsb_warmup.{pdf,png,csv}.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

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
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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
OUT_BASE = os.path.join(FIGURES_DIR, "ycsb_warmup")

NTX_PER_EPOCH = 100_000
SKEWS = ["0.01", "0.2", "0.4", "0.6", "0.8", "0.99"]

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\].*Running epoch (\d+)"
)
EVICT_RE = re.compile(r"Evictions needed:")
FILENAME_RE = re.compile(
    r"1KB_hybrid_staging_skew(?P<skew>[\d.]+)_rep(?P<rep>\d+)\.log$"
)


def parse_log(log_path: str):
    """Return (per_epoch_mtxn dict, first_eviction_epoch or None)."""
    times: Dict[int, datetime] = {}
    current_epoch: Optional[int] = None
    first_evict: Optional[int] = None
    with open(log_path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                current_epoch = int(m.group(3))
                ts = datetime.strptime(
                    f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S.%f"
                )
                times[current_epoch] = ts
                continue
            if first_evict is None and EVICT_RE.search(line) and current_epoch is not None:
                first_evict = current_epoch

    eps = sorted(times.keys())
    per_epoch: Dict[int, float] = {}
    for i in range(len(eps) - 1):
        dt = (times[eps[i + 1]] - times[eps[i]]).total_seconds()
        if dt > 0:
            per_epoch[eps[i]] = NTX_PER_EPOCH / dt / 1e6
    return per_epoch, first_evict


def collect():
    by_skew_tput: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    by_skew_evict: Dict[str, List[int]] = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "1KB_hybrid_staging_skew*.log"))):
        m = FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        skew = m.group("skew")
        per_epoch, first_evict = parse_log(log)
        for ep, mt in per_epoch.items():
            by_skew_tput[skew][ep].append(mt)
        if first_evict is not None:
            by_skew_evict[skew].append(first_evict)
    return by_skew_tput, by_skew_evict


def render(by_skew_tput, by_skew_evict) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    if not by_skew_tput:
        raise SystemExit("no warmup data parsed; did record_size cells run?")

    fig, ax = plt.subplots(figsize=(9.0, 4.5))

    cmap = get_cmap("viridis")
    n_lines = len(SKEWS)
    for i, skew in enumerate(SKEWS):
        if skew not in by_skew_tput:
            continue
        eps = sorted(by_skew_tput[skew].keys())
        medians = [statistics.median(by_skew_tput[skew][e]) for e in eps]
        color = cmap(i / max(1, n_lines - 1))
        ax.plot(eps, medians, color=color, linewidth=1.6,
                label=rf"$\theta={skew}$")

    # Eviction-onset vertical lines (thin red dashed). One per skew that
    # ever evicted; positioned at the median onset across 3 reps.
    for skew in SKEWS:
        evict_eps = by_skew_evict.get(skew, [])
        if not evict_eps:
            continue
        onset = int(statistics.median(evict_eps))
        ax.axvline(onset, color="#b03a2e", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("epoch", fontsize=13)
    ax.set_ylabel("Throughput (MTxn/s)", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle=":")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 1.18),
        ncol=6, fontsize=12, frameon=False,
    )
    ax.set_xlim(0, 305)
    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["skew", "epoch", "median_mtxn_s", "n_reps"])
        for skew in SKEWS:
            if skew not in by_skew_tput:
                continue
            for ep in sorted(by_skew_tput[skew].keys()):
                vs = by_skew_tput[skew][ep]
                w.writerow([skew, ep, round(statistics.median(vs), 3), len(vs)])
        w.writerow([])
        w.writerow(["skew", "eviction_onset_epoch_median", "n_reps"])
        for skew in SKEWS:
            evict_eps = by_skew_evict.get(skew, [])
            if not evict_eps:
                w.writerow([skew, "never", 0]); continue
            w.writerow([skew, int(statistics.median(evict_eps)), len(evict_eps)])
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(f"log dir not found: {LOG_DIR}; run 03_record_size.sh first")
    by_skew_tput, by_skew_evict = collect()
    render(by_skew_tput, by_skew_evict)


if __name__ == "__main__":
    main()
