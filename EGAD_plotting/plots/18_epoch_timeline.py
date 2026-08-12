#!/usr/bin/env python3
"""Paper figure 18: measured epoch timeline, async vs sync writeback.

Two panels from the fig-14 writeback_breakdown YCSB-F logs (300 epochs,
no new runs, no added instrumentation; every bar comes from log lines the
driver already prints).

  (a) async: two lanes. The main thread runs the epoch pipeline; the
      writeback worker runs the previous epoch's writeback split into
      two halves around the admission gather (the cv-gated pause in
      the stager), so D2H+scatter rides under the epoch's other phases.
  (b) sync: one lane. The same writeback runs inline after execution
      and stretches the epoch.

Epoch choice: the median-duration epoch in the steady window [250,280]
of rep1, so the panel shows a typical epoch, not a best case.

Reads $EGAD_LOGS_DIR/writeback_breakdown/ycsbf_{async,sync}_rep1.log;
writes $EGAD_FIGURES_DIR/epoch_timeline.{pdf,png,csv}.
"""
from __future__ import annotations

import csv
import os
import re
import statistics
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType embedding for paper-quality PDFs
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"  # match the paper body text
matplotlib.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["xtick.direction"] = "out"
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
OUT_BASE = os.path.join(FIGURES_DIR, "epoch_timeline")

WINDOW = (250, 280)
N_EPOCHS_SHOWN = 2

TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)\] \[\w\] \[thread (\d+)\]")
PHASE_RE = re.compile(r"Epoch (\d+) ([a-z_0-9]+) time: (\d+)")
WORKER_RE = re.compile(r"worker_phases: half1=(\d+) sg_wait=(\d+) half2=(\d+) total=(\d+)")
TV_RE = re.compile(r"\[SG\] SG\.transfer_versions\.total: (\d+) us")

# Same palette as figure 14 (breakdown_combined) so the pair reads together.
COLORS = {
    "index_transfer": "#6B9F71",
    "indexing":       "#93B883",
    "submission":     "#C9A961",
    "staging":        "#1F4E79",
    "admission_transfer": "#5E93C5",
    "initialization": "#B8A8D5",
    "execution":      "#6B5390",
    "flush":          "#B23A48",   # sync inline writeback
    "wb_half":        "#B23A48",   # worker D2H+scatter halves
    "wb_handoff":     "#D98880",   # main-thread flush_start_async handoff
    "wb_wait":        "#DDDDDD",   # worker pauses for the admission gather
}
LEGEND = [
    ("index_transfer", "Index transfer"),
    ("indexing", "Indexing"),
    ("submission", "Submit"),
    ("staging", "Staging (select + build)"),
    ("admission_transfer", "Admission transfer (gather + H2D)"),
    ("initialization", "MVCC init"),
    ("execution", "Execution"),
    ("wb_handoff", "Writeback handoff"),
    ("wb_half", "Writeback (D2H + scatter)"),
    ("wb_wait", "Worker paused (admission gather)"),
]
# In-bar labels for wide blocks only.
BAR_TEXT = {
    "staging": "staging",
    "admission_transfer": "admission transfer",
    "execution": "exec",
    "flush": "writeback",
    "initialization": "init",
}


def us_of(datestr: str, timestr: str) -> float:
    dt = datetime.strptime(f"{datestr} {timestr}", "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp() * 1e6


def parse(path):
    phases, workers, transfers = [], [], []
    with open(path) as f:
        for line in f:
            m = TS_RE.match(line)
            if not m:
                continue
            t, tid = us_of(m.group(1), m.group(2)), int(m.group(3))
            pm = PHASE_RE.search(line)
            if pm:
                phases.append((int(pm.group(1)), pm.group(2), int(pm.group(3)), t))
            wm = WORKER_RE.search(line)
            if wm:
                h1, wait, h2, tot = map(int, wm.groups())
                workers.append((h1, wait, h2, tot, t))
            tm = TV_RE.search(line)
            if tm:
                transfers.append((t - int(tm.group(1)), int(tm.group(1)), tid))
    return phases, workers, transfers


def epoch_start(phases, e):
    for ep, name, dur, end in phases:
        if ep == e and name == "index_transfer":
            return end - dur
    raise KeyError(f"epoch {e} has no index_transfer line")


def median_epoch(phases):
    lo, hi = WINDOW
    tot = {}
    for e in range(lo, hi + 1):
        try:
            tot[e] = epoch_start(phases, e + 1) - epoch_start(phases, e)
        except KeyError:
            continue
    med = statistics.median(tot.values())
    return min(tot, key=lambda e: abs(tot[e] - med)), tot


def fig14_total_ms(mode: str) -> float:
    """Per-epoch total computed exactly as figure 14 does (per-rep mean of
    summed phase durations over the window, then mean across reps), so the
    two figures quote identical numbers."""
    import glob
    WB = ("flush", "flush_sync_prev", "flush_start_async")
    rep_means = []
    for log in sorted(glob.glob(os.path.join(LOG_DIR, f"ycsbf_{mode}_rep*.log"))):
        per_epoch = {}
        for ep, name, dur, _end in parse(log)[0]:
            if WINDOW[0] <= ep <= WINDOW[1] and (name in COLORS or name in WB):
                per_epoch.setdefault(ep, 0)
                per_epoch[ep] += dur
        if per_epoch:
            rep_means.append(statistics.mean(per_epoch.values()))
    return statistics.mean(rep_means) / 1000.0


def main_segments(phases, transfers, t0, t1):
    """(start, dur, kind) for pipeline phases whose span lies in [t0, t1].

    The staging block is split around the admission transfer
    (transfer_versions), the sub-span whose gather the writeback worker
    waits out, so the pause and the transfer line up visibly."""
    segs = []
    for ep, name, dur, end in phases:
        start = end - dur
        if start < t0 - 500 or end > t1 + 2500 or dur == 0:
            continue
        if name == "flush_start_async":
            segs.append((start, dur, "wb_handoff"))
        elif name == "staging":
            tv = [(s, d) for s, d, _tid in transfers
                  if start <= s and s + d <= end + 200]
            if tv:
                ts, td = max(tv, key=lambda x: x[1])
                segs.append((start, ts - start, "staging"))
                segs.append((ts, td, "admission_transfer"))
                if end - (ts + td) > 50:
                    segs.append((ts + td, end - (ts + td), "staging"))
            else:
                segs.append((start, dur, "staging"))
        elif name in COLORS:
            segs.append((start, dur, name))
    return segs


def worker_segments(workers, t0, t1):
    """Each worker bar is one epoch's writeback: half1, pause, half2."""
    bars = []
    for h1, wait, h2, tot, end in workers:
        s = end - tot
        if end < t0 or s > t1:
            continue
        bars.append([(s, h1, "wb_half"), (s + h1, wait, "wb_wait"),
                     (s + h1 + wait, h2, "wb_half")])
    return bars


def draw(ax, y, segs, t0, height=0.6, label_big=True):
    for start, dur, kind in segs:
        x = (start - t0) / 1000.0
        w = dur / 1000.0
        ax.barh(y, w, left=x, height=height, color=COLORS[kind],
                edgecolor="white", linewidth=0.4, zorder=3)
        txt = BAR_TEXT.get(kind)
        if label_big and txt and w > 1.1:
            ax.text(x + w / 2, y, txt, ha="center", va="center",
                    fontsize=11, color="white", zorder=4)


def render():
    ap, aw, atv = parse(os.path.join(LOG_DIR, "ycsbf_async_rep1.log"))
    sp, _, stv = parse(os.path.join(LOG_DIR, "ycsbf_sync_rep1.log"))

    ae, atot = median_epoch(ap)
    se, stot = median_epoch(sp)
    a_ms = fig14_total_ms("async")
    s_ms = fig14_total_ms("sync")
    print(f"async median epoch e{ae} ({atot[ae]/1000:.2f} ms); window median {a_ms:.2f}")
    print(f"sync  median epoch e{se} ({stot[se]/1000:.2f} ms); window median {s_ms:.2f}")

    at0 = epoch_start(ap, ae)
    at1 = epoch_start(ap, ae + N_EPOCHS_SHOWN)
    st0 = epoch_start(sp, se)
    st1 = epoch_start(sp, se + N_EPOCHS_SHOWN)

    a_main = main_segments(ap, atv, at0, at1)
    a_workers = worker_segments(aw, at0, at1)
    s_main = main_segments(sp, stv, st0, st1)

    xmin = min([0.0] + [(b[0][0] - at0) / 1000.0 for b in a_workers]) - 0.15
    xmax = max((at1 - at0), (st1 - st0)) / 1000.0 + 0.3

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13.5, 4.6), sharex=True,
        gridspec_kw=dict(height_ratios=[2.0, 1.15], hspace=0.52),
    )

    # (a) async: main lane + worker lane
    draw(ax1, 1.0, a_main, at0)
    for i, bar in enumerate(a_workers):
        draw(ax1, 0.0, bar, at0)
        s = (bar[0][0] - at0) / 1000.0
        e = (bar[-1][0] + bar[-1][1] - at0) / 1000.0
        which = "$e{-}1$" if i == 0 else "$e$" if i == 1 else f"$e{{+}}{i-1}$"
        ax1.text((s + e) / 2, -0.62, f"writeback of epoch {which}",
                 ha="center", va="top", fontsize=11)
    ax1.set_yticks([1.0, 0.0])
    ax1.set_yticklabels(["Main\nthread", "Writeback\nworker"], fontsize=13)
    ax1.set_ylim(-1.05, 1.62)
    for k in range(N_EPOCHS_SHOWN + 1):
        x = (epoch_start(ap, ae + k) - at0) / 1000.0
        ax1.axvline(x, color="0.45", linestyle="--", linewidth=0.9, zorder=2)
    for k in range(N_EPOCHS_SHOWN):
        xm = ((epoch_start(ap, ae + k) + epoch_start(ap, ae + k + 1)) / 2 - at0) / 1000.0
        lab = "$e$" if k == 0 else f"$e{{+}}{k}$"
        ax1.text(xm, 1.52, f"epoch {lab}", ha="center", va="bottom", fontsize=12)
    # Dotted marker at each admission transfer's end, through both lanes:
    # the worker's pause terminates exactly here (the gather-done signal).
    for start, dur, kind in a_main:
        if kind == "admission_transfer":
            ax1.axvline((start + dur - at0) / 1000.0, color="#5E93C5",
                        linestyle=":", linewidth=1.4, zorder=2)
    ax1.set_title(rf"(a) Asynchronous writeback, {a_ms:.2f} ms per epoch",
                  fontsize=14, pad=30)

    # (b) sync: single lane
    draw(ax2, 0.0, s_main, st0)
    ax2.set_yticks([0.0])
    ax2.set_yticklabels(["Main\nthread"], fontsize=13)
    ax2.set_ylim(-0.62, 0.95)
    for k in range(N_EPOCHS_SHOWN + 1):
        x = (epoch_start(sp, se + k) - st0) / 1000.0
        ax2.axvline(x, color="0.45", linestyle="--", linewidth=0.9, zorder=2)
    for k in range(N_EPOCHS_SHOWN):
        xm = ((epoch_start(sp, se + k) + epoch_start(sp, se + k + 1)) / 2 - st0) / 1000.0
        lab = "$e$" if k == 0 else f"$e{{+}}{k}$"
        ax2.text(xm, 0.62, f"epoch {lab}", ha="center", va="bottom", fontsize=12)
    ax2.set_title(rf"(b) Synchronous writeback, {s_ms:.2f} ms per epoch",
                  fontsize=14, pad=24)
    ax2.set_xlabel("Time from start of epoch $e$ (ms)", fontsize=14)

    ax1.set_xlim(xmin, xmax)
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=12)
        ax.grid(True, axis="x", alpha=0.25, linestyle=":")

    handles = [mpatches.Patch(facecolor=COLORS[k], label=lab) for k, lab in LEGEND]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.20),
                     ncol=5, fontsize=11.5, frameon=False, handlelength=1.4,
                     columnspacing=1.2)

    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight", bbox_extra_artists=[leg])
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=300, bbox_extra_artists=[leg])
    print(f"saved {OUT_BASE}.{{pdf,png}}")

    with open(OUT_BASE + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "lane", "kind", "start_ms", "dur_ms"])
        for start, dur, kind in sorted(a_main):
            w.writerow(["async", "main", kind,
                        round((start - at0) / 1000.0, 3), round(dur / 1000.0, 3)])
        for bar in a_workers:
            for start, dur, kind in bar:
                w.writerow(["async", "worker", kind,
                            round((start - at0) / 1000.0, 3), round(dur / 1000.0, 3)])
        for start, dur, kind in sorted(s_main):
            w.writerow(["sync", "main", kind,
                        round((start - st0) / 1000.0, 3), round(dur / 1000.0, 3)])
        w.writerow(["async", "epoch_ms", "median_window", round(a_ms, 2), ""])
        w.writerow(["sync", "epoch_ms", "median_window", round(s_ms, 2), ""])
    print(f"saved {OUT_BASE}.csv")


if __name__ == "__main__":
    if not os.path.isdir(LOG_DIR):
        raise SystemExit(f"log dir not found: {LOG_DIR}; run 08_writeback_breakdown.sh first")
    render()
