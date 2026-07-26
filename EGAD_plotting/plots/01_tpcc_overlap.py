#!/usr/bin/env python3
"""Paper TPCC overlap figure.

Renders one steady-state TPC-C tpccdeck epoch as three stacked bands:

  * main thread     -- one bar with every epoch phase labeled in-block.
  * PE per-stager   -- a uniform light-grey envelope per stager with only
                       the SG transfers (admit_sg_xfer, evict:sg_transfer)
                       overlaid in color.
  * WB per-stager   -- async writeback worker bars, X-cropped at ~1.5x the
                       epoch duration. Workers from the previous epoch are
                       rendered in a faded style; workers spawned this
                       epoch are solid.

Reads $EGAD_LOGS_DIR/tpcc_overlap.log; writes $EGAD_FIGURES_DIR/tpcc_overlap.{pdf,png}.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
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
from matplotlib.transforms import blended_transform_factory

LOGS_DIR = os.environ.get(
    "EGAD_LOGS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"),
)
FIGURES_DIR = os.environ.get(
    "EGAD_FIGURES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"),
)
os.makedirs(FIGURES_DIR, exist_ok=True)

LOG_PATH = os.path.join(LOGS_DIR, "tpcc_overlap.log")
OUT_BASE = os.path.join(FIGURES_DIR, "tpcc_overlap")

# Steady-state eviction-path epoch.
TARGET_EPOCH = 46

# Stager render order (top -> bottom).
STAGERS = [
    "warehouse",
    "district",
    "customer",
    "item",
    "stock",
    "new_order",
    "order",
    "order_line",
]

# ===== Paper palette =====
# Phases are grouped by family (indexing / submit / PE / init / exec /
# async-writeback) with muted shades inside each family. Shared between the
# TPCC and YCSB overlap plots.

# Indexing family (greens):
C_INDEX_TRANSFER = "#6B9F71"   # medium green
C_AUX_INDEX      = "#B0C9A2"   # pale green (TPCC only)
C_INDEXING       = "#93B883"   # light-medium green

# Submission (mustard, distinct):
C_SUBMISSION     = "#C9A961"

# prepareEpoch family (blues; deeper = transfer, palest = rebuild tail):
C_PE_PARENT      = "#4A78AE"   # TPCC prepareEpoch parent block
C_PE_ENVELOPE    = "#E8E8E8"   # transfer-computations envelope (TPCC)
C_PE_ADMIT_SG    = "#2D5489"   # admit-side scatter-gather transfer
C_PE_EVICT_SG    = "#1F3A6B"   # evict-side scatter-gather transfer

# Init / exec family (purples; init lighter, exec deeper):
C_INIT           = "#B8A8D5"   # initialization (light lavender)
C_EXEC           = "#6B5390"   # execution (deeper royal purple)

# Async / writeback family (reds; better contrast with the blue PE):
C_SYNC_FLUSH     = "#D9A39C"   # sync flush (light muted red)
C_START_FLUSH    = "#B26B5E"   # start flush async (medium muted red)
C_WB_THIS        = "#8C3D33"   # writeback worker spawned this epoch (deep red)
C_WB_PREV        = "#ECC9C3"   # writeback worker spawned previous epoch (very pale red)

# Main-thread phases in chronological order: (label_for_plot, log_phase_name, color).
MAIN_PHASES = [
    ("index transfer",     "index_transfer",                     C_INDEX_TRANSFER),
    # `cpu aux index` is a zero-duration marker emitted BEFORE index_transfer
    # and is dropped from the visualization. The two GPU aux phases ARE
    # contiguous and get merged into one block by render's same-label collapse.
    ("auxiliary index",    "gpu aux index",                      C_AUX_INDEX),
    ("auxiliary index",    "gpu aux index part2",                C_AUX_INDEX),
    ("indexing",           "indexing",                           C_INDEXING),
    ("submission",         "submission",                         C_SUBMISSION),
    ("prepareEpoch",       "hybrid_staging:prepareEpoch",        C_PE_PARENT),
    ("initialization",     "initialization",                     C_INIT),
    ("execution",          "execution",                          C_EXEC),
    ("sync flush",         "hybrid_staging:flush_sync_prev",     C_SYNC_FLUSH),
    ("start flush async",  "hybrid_staging:flush_start_async",   C_START_FLUSH),
]

PE_ENVELOPE_COLOR = C_PE_ENVELOPE
ADMIT_SG_COLOR    = C_PE_ADMIT_SG
EVICT_SG_COLOR    = C_PE_EVICT_SG

WB_THIS_EPOCH_COLOR = C_WB_THIS
WB_PREV_EPOCH_COLOR = C_WB_PREV

EPOCHS_AFTER_TARGET_TO_SHOW = 0.25  # crop X axis at this fraction of one epoch past the target's end


# ---------- log parsing ----------

TIMESTAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)\]")


def parse_ts_us(line: str) -> Optional[int]:
    m = TIMESTAMP_RE.search(line)
    if not m:
        return None
    h, mi, s, frac = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    return ((h * 3600 + mi * 60 + s) * 1_000_000) + frac


@dataclass
class EpochData:
    main_phase_end_us: Dict[str, int] = field(default_factory=dict)        # phase_name -> end timestamp
    main_phase_dur_us: Dict[str, int] = field(default_factory=dict)        # phase_name -> duration
    pe_subphase_offsets: Dict[str, Dict[str, int]] = field(default_factory=dict)  # stager -> marker -> offset_us
    pe_total_us: Dict[str, int] = field(default_factory=dict)              # stager -> total PE duration
    pe_start_us: Dict[str, int] = field(default_factory=dict)              # stager -> wallclock PE start
    sfa_start_us: Dict[str, int] = field(default_factory=dict)             # stager -> wallclock per-stager start_flush_async start
    sfa_end_us: Dict[str, int] = field(default_factory=dict)               # stager -> wallclock per-stager start_flush_async end
    epoch_start_us: Optional[int] = None


@dataclass
class WorkerEvent:
    end_ts_us: int
    addr: str
    total_us: int
    spawn_epoch: Optional[int] = None  # filled in after parse via FIFO pairing


def parse_log(path: str) -> Tuple[Dict[int, EpochData], Dict[str, str], List[WorkerEvent]]:
    """Returns (epochs, addr_to_stager, workers).

    epochs[ep_id]: EpochData
    addr_to_stager: 0x... -> 'order_line'
    workers: list of WorkerEvent. Each worker's spawn_epoch is paired FIFO
    with the per-stager start_flush_async events: the n-th worker_phases
    log line for an addr was spawned by the n-th start_flush_async event=end
    for that addr. This is more accurate than classifying by work-start
    timestamp, since CV wakeup latency can let a worker's actual work begin
    a few hundred us into the next epoch on the wall clock.
    """
    epochs: Dict[int, EpochData] = defaultdict(EpochData)
    addr_to_stager: Dict[str, str] = {}
    workers: List[WorkerEvent] = []
    spawn_epoch_by_addr: Dict[str, List[int]] = defaultdict(list)  # addr -> [epoch, epoch, ...] in order

    inv_pe_start = re.compile(
        r"\[INV-TL\] epoch=(\d+) stager=(\w+) phase=prepareEpoch event=start")
    inv_sfa_start = re.compile(
        r"\[INV-TL\] epoch=(\d+) stager=(\w+) phase=start_flush_async event=start")
    inv_sfa_end = re.compile(
        r"\[INV-TL\] epoch=(\d+) stager=(\w+) phase=start_flush_async event=end "
        r".*?stager_addr=(0x[0-9a-fA-F]+|\(nil\))")
    epoch_phase = re.compile(r"Epoch (\d+) (.+?) time: (\d+) us\b")
    tl_line = re.compile(r"\[TL\] epoch=(\d+)\s*([\w@\d\s|]+)")
    worker_phases = re.compile(
        r"\[HYBRID\] worker_phases: total=(\d+)(?: addr=(0x[0-9a-fA-F]+|\(nil\)))?")

    # The [TL] markers we keep (admit + evict SG). All other PE sub-phases dropped.
    sg_markers = {"admit_sg_xfer_start", "sg_transfer_start", "sg_transfer_end"}

    # Parse [TL] line into stager identity. The [TL] line itself doesn't carry
    # stager= directly (older format); we must associate by the preceding
    # INV-TL prepareEpoch start on the same epoch+thread. We use a
    # thread-tagged accumulator.
    last_pe_start_for_thread: Dict[str, Tuple[int, str]] = {}  # thread_id -> (epoch, stager)

    with open(path) as f:
        for line in f:
            ts = parse_ts_us(line)

            m = inv_pe_start.search(line)
            if m and ts is not None:
                ep, stager = int(m.group(1)), m.group(2)
                epochs[ep].pe_start_us[stager] = ts
                if epochs[ep].epoch_start_us is None and stager == "warehouse":
                    epochs[ep].epoch_start_us = ts
                # Track for [TL] association.
                tid_m = re.search(r"\[thread (\d+)\]", line)
                if tid_m:
                    last_pe_start_for_thread[tid_m.group(1)] = (ep, stager)
                continue

            m = inv_sfa_start.search(line)
            if m and ts is not None:
                ep, stager = int(m.group(1)), m.group(2)
                epochs[ep].sfa_start_us[stager] = ts
                continue

            m = inv_sfa_end.search(line)
            if m:
                ep, stager, addr = int(m.group(1)), m.group(2), m.group(3)
                addr_to_stager[addr] = stager
                if ts is not None:
                    epochs[ep].sfa_end_us[stager] = ts
                # Record the spawn epoch for this addr (used to pair with the
                # corresponding worker_phases log line FIFO).
                spawn_epoch_by_addr[addr].append(ep)
                continue

            m = epoch_phase.search(line)
            if m and ts is not None:
                ep = int(m.group(1))
                phase, dur = m.group(2).strip(), int(m.group(3))
                epochs[ep].main_phase_end_us[phase] = ts
                epochs[ep].main_phase_dur_us[phase] = dur
                continue

            m = tl_line.search(line)
            if m:
                ep = int(m.group(1))
                tid_m = re.search(r"\[thread (\d+)\]", line)
                if not tid_m:
                    continue
                tid = tid_m.group(1)
                tagged_ep, stager = last_pe_start_for_thread.get(tid, (None, None))
                if tagged_ep != ep or stager is None:
                    continue
                # Parse marker offsets out of the body.
                body = m.group(2)
                # Tokens look like:  start@0 |dur1| label1@off1 |dur2| label2@off2 ...
                # We only care about offsets for the markers in sg_markers, plus 'end' for total.
                offs: Dict[str, int] = {}
                for mm in re.finditer(r"(\w+)@(\d+)", body):
                    name, off = mm.group(1), int(mm.group(2))
                    offs[name] = off
                epochs[ep].pe_subphase_offsets[stager] = offs
                if "end" in offs:
                    epochs[ep].pe_total_us[stager] = offs["end"]
                continue

            m = worker_phases.search(line)
            if m and ts is not None:
                total = int(m.group(1))
                addr = m.group(2)
                if addr is None:
                    continue
                workers.append(WorkerEvent(end_ts_us=ts, addr=addr, total_us=total))
                continue

    # Pair each worker with its spawn epoch FIFO per addr.
    consumed_per_addr: Dict[str, int] = defaultdict(int)
    for w in workers:
        spawns = spawn_epoch_by_addr.get(w.addr, [])
        idx = consumed_per_addr[w.addr]
        if idx < len(spawns):
            w.spawn_epoch = spawns[idx]
            consumed_per_addr[w.addr] = idx + 1
        # else: more workers than spawns — leave spawn_epoch=None (renders as faded).

    return epochs, addr_to_stager, workers


# ---------- rendering ----------

def fmt_us(us: int) -> str:
    if us >= 1000:
        return f"{us/1000:.1f}ms"
    return f"{us}us"


def _is_light(hex_color: str) -> bool:
    """Return True if the color is light enough that black text reads better."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) > 160


def add_bar_label(ax, x_left_ms: float, width_ms_v: float, y_center: float,
                  label: str, dur_us: int, color: str,
                  inches_per_ms: float) -> None:
    """Centered in-block label that fits the bar.

    The duration-threshold rule (8/7/6/5) sets an upper bound on fontsize;
    from there we scan down and at each level try the label as-is, then
    split-on-first-space if it's multi-word. The first variant whose
    longest line fits in 88% of the bar width wins. If nothing fits or the
    bar is <= 50 us, the label is skipped.
    """
    if dur_us <= 50:
        return
    if dur_us > 1500:
        base_fs = 8
    elif dur_us > 400:
        base_fs = 7
    elif dur_us > 150:
        base_fs = 6
    else:
        base_fs = 5

    cap_in = width_ms_v * inches_per_ms * 0.88

    def width_in(text: str, fs: int) -> float:
        longest = max(text.split('\n'), key=len)
        return len(longest) * 0.58 * fs / 72

    chosen = None
    for fs in range(base_fs, 2, -1):
        if width_in(label, fs) <= cap_in:
            chosen = (label, fs)
            break
        if ' ' in label:
            split = '\n'.join(label.split(' ', 1))
            if width_in(split, fs) <= cap_in:
                chosen = (split, fs)
                break

    if chosen is None:
        return

    text_v, fs = chosen
    weight = "bold" if (dur_us > 1500 and fs == 8 and '\n' not in text_v) else "normal"
    text_color = "black" if _is_light(color) else "white"
    cx = x_left_ms + width_ms_v / 2
    ax.text(cx, y_center, text_v, ha="center", va="center",
            fontsize=fs, color=text_color, fontweight=weight,
            clip_on=True, zorder=5)


def render(target_epoch: int, epochs: Dict[int, EpochData],
           addr_to_stager: Dict[str, str], workers: List[WorkerEvent]) -> None:
    # Delete any prior outputs of this plot so the figures dir never
    # accumulates stale variants.
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    if target_epoch not in epochs:
        raise SystemExit(f"target epoch {target_epoch} not present in log")
    if target_epoch + 1 not in epochs:
        raise SystemExit(f"need epoch {target_epoch+1} for X-axis cropping")

    e = epochs[target_epoch]
    e_next = epochs[target_epoch + 1]
    if e.epoch_start_us is None or e_next.epoch_start_us is None:
        raise SystemExit("epoch_start markers missing")

    # Anchor t=0 at the earliest main-thread phase start of this epoch so the
    # pre-prepareEpoch phases (index_transfer, indexing, submission, ...)
    # are not chopped off.
    main_phase_starts = [
        e.main_phase_end_us[k] - e.main_phase_dur_us[k]
        for _, k, _ in MAIN_PHASES
        if k in e.main_phase_end_us
    ]
    if not main_phase_starts:
        raise SystemExit("no main-thread phases found for target epoch")
    t0 = min(main_phase_starts)
    epoch_dur_us = e_next.epoch_start_us - t0
    x_max_us = epoch_dur_us * (1.0 + EPOCHS_AFTER_TARGET_TO_SHOW)

    # Approximate axes width in inches after tight_layout; used by
    # add_bar_label to size labels so they fit their bar. 0.92 of fig width
    # is a reliable post-tight_layout estimate for our 15-inch wide plot.
    inches_per_ms = (15 * 0.92) / (x_max_us / 1000.0)

    def to_ms(us_abs: int) -> float:
        return (us_abs - t0) / 1000.0
    def width_ms(us: int) -> float:
        return us / 1000.0

    # WB rows are ordered by per-stager start_flush_async start time this
    # epoch (spawn order), so the row at the top is the first one submitted
    # in start_flush_async (typically order_line per the dispatch in tpcc.cpp).
    wb_order = sorted(
        [s for s in STAGERS if s in e.sfa_start_us],
        key=lambda s: e.sfa_start_us[s],
    )
    # PE rows stay in fixed canonical order.
    pe_order = list(STAGERS)

    # Collapsed layout: 3 logical rows (main + PE + WB). PE and WB each show
    # ONE bar -- the order_line stager as the representative -- and the
    # "x 8 tables" y-axis label tells the reader that the row stands in for
    # 8 per-table stagers running in parallel.
    REPRESENTATIVE_STAGER = "order_line"
    row_h = 0.55          # bar height (shared across the 3 rows)
    sub_row_h = 0.55      # PE / WB bar height (matches main since one bar each)
    row_gap = 0.45        # gap between rows
    row_pitch = row_h + row_gap

    wb_y   = row_h / 2
    pe_y   = wb_y + row_pitch
    main_y = pe_y + row_pitch

    fig_height = 2.4
    fig, ax = plt.subplots(figsize=(15, fig_height))

    # ---- Main row (in-block labels) ----
    # Phases that share a label (e.g. the three "aux index" sub-phases) are
    # collapsed into a single contiguous bar so the rendered block has no
    # internal seams between adjacent same-color segments.
    main_segments_by_label: Dict[str, List[Tuple[int, int, str]]] = {}
    for label, phase_key, color in MAIN_PHASES:
        if phase_key not in e.main_phase_end_us:
            continue
        end_us = e.main_phase_end_us[phase_key]
        dur_us = e.main_phase_dur_us[phase_key]
        start_us = end_us - dur_us
        main_segments_by_label.setdefault(label, []).append((start_us, end_us, color))
    for label, segments in main_segments_by_label.items():
        s_min = min(s for s, _, _ in segments)
        e_max = max(en for _, en, _ in segments)
        color = segments[0][2]
        if s_min < t0:
            continue
        x0 = to_ms(s_min)
        w  = width_ms(e_max - s_min)
        ax.barh(main_y, w, left=x0, height=row_h, color=color, edgecolor="black", linewidth=0.4)
        add_bar_label(ax, x0, w, main_y, label, e_max - s_min, color, inches_per_ms)

    # ---- PE row: single bar showing the order_line stager's PE envelope
    # plus its admit / evict SG transfers. The y-axis "x 8 tables" annotation
    # tells the reader 7 other per-table stagers run in parallel.
    s = REPRESENTATIVE_STAGER
    if s in e.pe_start_us and s in e.pe_total_us:
        pe_start_abs = e.pe_start_us[s]
        pe_total = e.pe_total_us[s]
        x0 = to_ms(pe_start_abs)
        w  = width_ms(pe_total)
        ax.barh(pe_y, w, left=x0, height=sub_row_h, color=PE_ENVELOPE_COLOR,
                edgecolor="#bbbbbb", linewidth=0.3)

        offs = e.pe_subphase_offsets.get(s, {})
        if "admit_sg_xfer_start" in offs:
            start_off = offs["admit_sg_xfer_start"]
            later = [v for v in offs.values() if v > start_off]
            end_off = min(later) if later else pe_total
            sg_dur = end_off - start_off
            sg_x0 = to_ms(pe_start_abs + start_off)
            sg_w = width_ms(sg_dur)
            ax.barh(pe_y, sg_w, left=sg_x0, height=sub_row_h,
                    color=ADMIT_SG_COLOR)
            add_bar_label(ax, sg_x0, sg_w, pe_y, "admit transfer", sg_dur,
                          ADMIT_SG_COLOR, inches_per_ms)
        if "sg_transfer_start" in offs and "sg_transfer_end" in offs:
            sg_dur = offs["sg_transfer_end"] - offs["sg_transfer_start"]
            sg_x0 = to_ms(pe_start_abs + offs["sg_transfer_start"])
            sg_w = width_ms(sg_dur)
            ax.barh(pe_y, sg_w, left=sg_x0, height=sub_row_h,
                    color=EVICT_SG_COLOR)
            add_bar_label(ax, sg_x0, sg_w, pe_y, "evict transfer", sg_dur,
                          EVICT_SG_COLOR, inches_per_ms)

    # ---- WB rows: per-stager submission segments + worker bars ----
    # Each WB row has two visual elements:
    #   1. The per-stager start_flush_async submission window (small teal
    #      segment matching the main row's start_async color), drawn from
    #      sfa_start_us to sfa_end_us. Tells the reader exactly when the
    #      main thread spent time submitting this stager's worker.
    #   2. The worker bar itself: SOLID green if the worker was spawned this
    #      epoch (FIFO pairing with start_flush_async event=end), FADED
    #      green if it is a leftover from the previous epoch.
    # This-epoch worker bar is visually shifted to start at the END of the
    # entire main-thread flush_start_async phase (clean separation from the
    # main-row "start flush async" block). Prev-epoch worker is rendered at
    # actual wall clock (end = w.end_ts_us, start = end - total_us) so its
    # tail correctly ends BEFORE this epoch's flush_sync_prev (which waits
    # on it). Shifting the prev bar would push its end past sync_prev.
    SFA_KEY = "hybrid_staging:flush_start_async"
    e_prev = epochs.get(target_epoch - 1)
    for w in workers:
        if w.addr not in addr_to_stager:
            continue
        stager = addr_to_stager[w.addr]
        if stager != REPRESENTATIVE_STAGER:
            continue

        if w.spawn_epoch == target_epoch:
            color = WB_THIS_EPOCH_COLOR
            if SFA_KEY not in e.main_phase_end_us:
                continue
            start_us = e.main_phase_end_us[SFA_KEY]
            end_us   = start_us + w.total_us
        elif w.spawn_epoch == target_epoch - 1 and e_prev is not None:
            color = WB_PREV_EPOCH_COLOR
            end_us   = w.end_ts_us
            start_us = end_us - w.total_us
        else:
            continue

        if end_us <= t0 or start_us >= t0 + x_max_us:
            continue
        vis_start = max(start_us, t0)
        vis_end   = min(end_us, t0 + x_max_us)
        x0 = to_ms(vis_start)
        wb_width = width_ms(vis_end - vis_start)
        ax.barh(wb_y, wb_width, left=x0, height=sub_row_h, color=color,
                edgecolor="black", linewidth=0.3, zorder=2)
        add_bar_label(ax, x0, wb_width, wb_y, "writeback",
                      vis_end - vis_start, color, inches_per_ms)

    # ---- Axis cosmetics ----
    ax.set_xlim(0, x_max_us / 1000.0)
    ax.set_ylim(-0.3, main_y + row_h / 2 + 0.3)
    ax.set_xlabel("ms (wall clock, t=0 at epoch start)")
    yticks = [main_y, pe_y, wb_y]
    yticklabels = ["main",
                   f"PE\n(× {len(pe_order)} tables)",
                   f"WB\n(× {len(wb_order)} tables)"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=12)
    ax.tick_params(axis="x", labelsize=12)

    # Vertical line at the next-epoch boundary; label inside the plot box at
    # the top, just to the right of the dashed line.
    ax.axvline(epoch_dur_us / 1000.0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.text(epoch_dur_us / 1000.0, 0.985, " next epoch",
            fontsize=14, color="#444", va="top",
            transform=blended_transform_factory(ax.transData, ax.transAxes))

    # No legend: in-block labels make it redundant.

    # ax.set_title(f"tpccdeck W=128 epoch {target_epoch} (steady state)", fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")


def main() -> None:
    if not os.path.exists(LOG_PATH):
        raise SystemExit(f"log not found: {LOG_PATH}; run tests first")
    epochs, addr_to_stager, workers = parse_log(LOG_PATH)
    render(TARGET_EPOCH, epochs, addr_to_stager, workers)


if __name__ == "__main__":
    main()
