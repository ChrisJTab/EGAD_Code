#!/usr/bin/env python3
"""Paper figure 06: config sensitivity for hybrid_staging.

Renders a skew sweep showing hybrid vs cpu_only at three (read, field-split)
combinations, all at 120 B + 20 M + ~33.6 % cache/DB ratio:

  -rT-fF  full-record read, no field-split             (headline)
  -rT-fT  full-record read, field-split storage        (falls short)
  -rF-fT  single-field read, field-split storage       (best case)

cache pressure is held identical (33.6 %) so the only thing varying across
configs is read/write semantics. The story: hybrid wins big at -rF-fT,
wins modestly at -rT-fF, and approaches / underperforms cpu_only at
-rT-fT because the 10x read amplification (full-record reads against
field-split storage) eats throughput.

Reads $EGAD_LOGS_DIR/config_sensitivity/*.log; writes
$EGAD_FIGURES_DIR/config_sensitivity.{pdf,png,csv}.
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
from matplotlib.lines import Line2D

LOGS_DIR = os.environ.get(
    "EGAD_LOGS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"),
)
FIGURES_DIR = os.environ.get(
    "EGAD_FIGURES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"),
)
os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_DIR = os.path.join(LOGS_DIR, "config_sensitivity")
OUT_BASE = os.path.join(FIGURES_DIR, "config_sensitivity")

NTX_PER_EPOCH = 100_000
HYBRID_WINDOW = (250, 280)
BASELINE_WINDOW = (3, 10)  # cpu_only: average over epochs 3-10 (steady-state, post-warmup)

# Color per config; solid = hybrid, dashed = cpu_only.
C_RTFF = "#1f4e79"   # headline blue
C_RTFT = "#b03a2e"   # falls-short red
C_RFFT = "#2e7d32"   # best-case green

CONFIG_LABEL = {
    "rTfF": "full read, no split",
    "rTfT": "full read, split",
    "rFfT": "single read, split",
}
CONFIG_COLOR = {"rTfF": C_RTFF, "rTfT": C_RTFT, "rFfT": C_RFFT}
CONFIG_ORDER = ["rTfF", "rTfT", "rFfT"]

EPOCH_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\].*Running epoch (\d+)"
)
FILENAME_RE = re.compile(
    r"(?P<bench>ycsb[abcf])_(?P<config>rTfF|rTfT|rFfT)_(?P<mode>hybrid_staging|cpu_only)_skew(?P<skew>[\d.]+)_rep(?P<rep>\d+)\.log$"
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
    by_cell: Dict[Tuple[str, str, float], List[float]] = defaultdict(list)
    for log in sorted(glob.glob(os.path.join(LOG_DIR, "*.log"))):
        m = FILENAME_RE.search(os.path.basename(log))
        if not m:
            continue
        config = m.group("config")
        mode = m.group("mode")
        skew = float(m.group("skew"))
        window = HYBRID_WINDOW if mode == "hybrid_staging" else BASELINE_WINDOW
        mt = steady_state_mtxn(log, window)
        if mt is not None:
            by_cell[(config, mode, skew)].append(mt)
    return by_cell


def render(cells: Dict[Tuple[str, str, float], List[float]]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    skews = sorted({k[2] for k in cells.keys()})
    if not skews:
        raise SystemExit("no cells parsed; did 06_config_sensitivity.sh run?")

    fig = plt.figure(figsize=(8.0, 4.8))
    ax = fig.add_axes([0.12, 0.15, 0.85, 0.65])

    for config in CONFIG_ORDER:
        color = CONFIG_COLOR[config]
        label_cfg = CONFIG_LABEL[config]
        for mode, ls, marker in (
            ("hybrid_staging", "-",  "o"),
            ("cpu_only",       "--", "x"),
        ):
            # Median center: the EPIC-CPU cells carry mode structure across
            # their reps (upstream's slow mode), and a mean would sit between
            # modes. Cells whose whiskers showed both modes carry ten reps.
            pts = []  # (skew, median, min, max)
            for s in skews:
                vs = cells.get((config, mode, s), [])
                if vs:
                    pts.append((s, statistics.median(vs), min(vs), max(vs)))
            if not pts:
                continue
            xs_plot = [p[0] for p in pts]
            ys_plot = [p[1] for p in pts]
            yerr = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
            ax.errorbar(
                xs_plot, ys_plot, yerr=yerr, capsize=2.5, elinewidth=1.0,
                color=color, linestyle=ls, marker=marker, markersize=8,
                linewidth=1.6,
            )

    ax.set_xlabel(r"Zipfian skew $\theta$", fontsize=17)
    ax.set_ylabel("Throughput (MTxn/s)", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # Split legend: color legend (3 configs) above the plot, line-style legend
    # (EGAD vs EPIC-CPU) inline in a free corner of the plot. Avoids the wide
    # 6-item compound legend that was squeezing the data area.
    color_handles = [
        Line2D([], [], color=CONFIG_COLOR[c], linewidth=2, label=CONFIG_LABEL[c])
        for c in CONFIG_ORDER
    ]
    style_handles = [
        Line2D([], [], color="black", linestyle="-",  marker="o", label="EGAD"),
        Line2D([], [], color="black", linestyle="--", marker="x", label="EPIC-CPU"),
    ]
    color_leg = ax.legend(
        handles=color_handles, loc="upper center", bbox_to_anchor=(0.5, 1.18),
        ncol=3, fontsize=16, frameon=False,
    )
    ax.add_artist(color_leg)
    ax.legend(
        handles=style_handles, loc="upper left", fontsize=15,
        frameon=True, framealpha=0.9, edgecolor="0.7",
    )
    # bbox_inches="tight" doesn't auto-include legends added via add_artist;
    # pass color_leg explicitly so it isn't cropped.
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight", bbox_extra_artists=[color_leg])
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600, bbox_extra_artists=[color_leg])
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    csv_path = OUT_BASE + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "mode", "skew", "median_mtxn_s", "min_mtxn_s", "max_mtxn_s", "n_reps"])
        for (config, mode, skew), vs in sorted(cells.items()):
            w.writerow([config, mode, skew, statistics.median(vs),
                        round(min(vs), 3), round(max(vs), 3), len(vs)])
    print(f"saved {csv_path}")


def main() -> None:
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(
            f"log dir not found: {LOG_DIR}; run 06_config_sensitivity.sh first"
        )
    cells = collect()
    render(cells)


if __name__ == "__main__":
    main()
