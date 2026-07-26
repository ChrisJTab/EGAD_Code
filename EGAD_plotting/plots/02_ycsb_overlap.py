#!/usr/bin/env python3
"""Paper YCSB-F (rF/fT/120 B) overlap figure with split writeback worker.

Renders one steady-state YCSB-F epoch as two bands:

  * main thread  - the full main-thread phase sequence, with the
                   prepareEpoch ("staging") block expanded inline into
                   its [TL] sub-phases (build / fifo:mark / fifo:select /
                   fifo:evict / bulk_update / update_resident / pre-SG
                   D2H / SG transfer / wipe / rebuild / promote / remap).
  * WB worker    - the async writeback worker, drawn as the
                   half1 / sg_wait / half2 split exposed by YCSB's
                   writeback design (one writeback call per half, with
                   an explicit sync wait in between so the second D2H
                   pipelines against the first scatter). Both the prev
                   epoch's tail and the current epoch's spawn are shown.

Color scheme and label style mirror plotting/plot_overlap_timeline.py.

Reads $EGAD_LOGS_DIR/ycsb_overlap.log; writes
$EGAD_FIGURES_DIR/ycsb_overlap.{pdf,png}.
"""
from __future__ import annotations

import os
import re
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
import matplotlib.patches as mpatches
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

LOG_PATH = os.path.join(LOGS_DIR, "ycsb_overlap.log")
OUT_BASE = os.path.join(FIGURES_DIR, "ycsb_overlap")

# Steady-state epoch (e250-280 per RUNNING_HYBRID_STAGER.md §4.1).
TARGET_EPOCH = 260

# ===== Paper palette =====
# Phases grouped by family with muted shades inside each family.
# Shared between the TPCC and YCSB overlap plots.

# Indexing family (greens):
C_INDEX_TRANSFER = "#6B9F71"
C_INDEXING       = "#93B883"

# Submission (mustard):
C_SUBMISSION     = "#C9A961"

# prepareEpoch sub-phase blue tiers (deepest = transfer, palest = rebuild tail).
C_PE_SG_XFER     = "#152C53"   # sg_transfer (deepest)
C_PE_D2H         = "#22497A"   # pre_sg_d2h
C_PE_COMPUTE_A   = "#345E96"   # build / compact
C_PE_COMPUTE_B   = "#476FA6"
C_PE_COMPUTE_C   = "#5A85B5"
C_PE_FIFO_A      = "#3E70A6"
C_PE_FIFO_B      = "#5688BA"
C_PE_FIFO_C      = "#74A3CC"
C_PE_DEDUP_A     = "#609BBE"   # dedup / map / classify / mark_dirty
C_PE_DEDUP_B     = "#7AAFCA"
C_PE_DEDUP_C     = "#92BFD3"
C_PE_DEDUP_D     = "#A6CCDB"
C_PE_REBUILD_A   = "#9CB7CD"   # wipe / rebuild / promote / remap (palest)
C_PE_REBUILD_B   = "#ABC2D4"
C_PE_REBUILD_C   = "#B8CCDA"
C_PE_REBUILD_D   = "#C5D4E0"

# Init / exec family (purples; init lighter, exec deeper):
C_INIT           = "#B8A8D5"   # initialization (light lavender)
C_EXEC           = "#6B5390"   # execution (deeper royal purple)

# Async / writeback family (reds; better contrast with the blue PE):
C_SYNC_FLUSH     = "#D9A39C"   # sync flush (light muted red)
C_START_FLUSH    = "#B26B5E"   # start flush async (medium muted red)
C_WB_HALF1_THIS  = "#B26B5E"   # this-epoch worker, half 1 (medium red)
C_WB_HALF2_THIS  = "#8C3D33"   # this-epoch worker, half 2 (deeper red)
C_WB_WAIT_THIS   = "#D5D5D5"   # this-epoch sg_wait (neutral gray gap)
C_WB_HALF1_PREV  = "#ECC9C3"   # prev-epoch half 1 (very pale red)
C_WB_HALF2_PREV  = "#D9A39C"   # prev-epoch half 2 (light red)
C_WB_WAIT_PREV   = "#ECECEC"   # prev-epoch sg_wait

# Pre-PE main-thread phases (chronological).
PRE_PE_PHASES = [
    ("index transfer", "index_transfer", C_INDEX_TRANSFER),
    ("indexing",       "indexing",       C_INDEXING),
    ("submission",     "submission",     C_SUBMISSION),
]
# Post-PE main-thread phases (chronological).
POST_PE_PHASES = [
    ("initialization",    "initialization",    C_INIT),
    ("execution",         "execution",         C_EXEC),
    ("sync\nflush",       "flush_sync_prev",   C_SYNC_FLUSH),
    ("start flush async", "flush_start_async", C_START_FLUSH),
]

# prepareEpoch sub-phases. [TL] marker name -> (label, color).
# All blue tones, with deeper shades for the transfer-heavy phases
# (pre_sg_d2h, sg_transfer) and the palest shades for the rebuild tail.
# Labels use full words (D2H is kept abbreviated; the FIFO prefix is dropped
# from mark/select/evict).
PE_SUBPHASE_CFG: Dict[str, Tuple[str, str]] = {
    "compact_start":         ("compact",                  C_PE_COMPUTE_A),
    "build_needed_start":    ("build\nneeded",            C_PE_COMPUTE_B),
    "build_loads_start":     ("build loads",              C_PE_COMPUTE_C),
    "build_write_start":     ("build write",              C_PE_COMPUTE_A),
    "fifo_mark_start":       ("mark",                     C_PE_FIFO_A),
    "fifo_select_start":     ("select",                   C_PE_FIFO_B),
    "fifo_evict_start":      ("evict",                    C_PE_FIFO_C),
    "bulk_update_start":     ("bulk update",              C_PE_COMPUTE_A),
    "update_resident_start": ("update resident",          C_PE_COMPUTE_C),
    "pre_sg_d2h_start":      ("D2H",                      C_PE_D2H),
    "sg_transfer_start":     ("scatter-gather transfer", C_PE_SG_XFER),
    "dedup_start":           ("deduplicate",              C_PE_DEDUP_A),
    "map_write_start":       ("map write",                C_PE_DEDUP_B),
    "classify_start":        ("classify",                 C_PE_DEDUP_C),
    "mark_dirty_start":      ("mark dirty",               C_PE_DEDUP_D),
    "wipe_start":            ("wipe",                     C_PE_REBUILD_A),
    "rebuild_start":         ("rebuild",                  C_PE_REBUILD_B),
    "promote_start":         ("promote",                  C_PE_REBUILD_C),
    "remap_start":           ("remap",                    C_PE_REBUILD_D),
}
# Markers that END a phase (no colored block drawn from them).
PE_END_MARKERS = {"start", "end", "sg_transfer_end"}

# Writeback split colors (purples; sg_wait stays a neutral gray to read as idle).
WB_HALF1_THIS = C_WB_HALF1_THIS
WB_WAIT_THIS  = C_WB_WAIT_THIS
WB_HALF2_THIS = C_WB_HALF2_THIS
WB_HALF1_PREV = C_WB_HALF1_PREV
WB_WAIT_PREV  = C_WB_WAIT_PREV
WB_HALF2_PREV = C_WB_HALF2_PREV

EPOCHS_AFTER_TARGET_TO_SHOW = 0.06   # crop X axis tightly past the next-epoch line


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
    main_phase_end_us: Dict[str, int] = field(default_factory=dict)
    main_phase_dur_us: Dict[str, int] = field(default_factory=dict)
    tl_offsets: Dict[str, int] = field(default_factory=dict)
    tl_total_us: Optional[int] = None
    pe_start_us: Optional[int] = None
    flush_async_end_us: Optional[int] = None


@dataclass
class Worker:
    end_ts_us: int
    half1_us: int
    sg_wait_us: int
    half2_us: int
    total_us: int


def parse_log(path: str) -> Tuple[Dict[int, EpochData], List[Worker]]:
    epochs: Dict[int, EpochData] = defaultdict(EpochData)
    workers: List[Worker] = []

    epoch_phase = re.compile(r"Epoch (\d+) ([\w_]+) time: (\d+) us\b")
    tl_line     = re.compile(r"\[TL\] epoch=(\d+)\s*(.+)")
    worker_re   = re.compile(
        r"\[HYBRID\] worker_phases: half1=(\d+) sg_wait=(\d+) half2=(\d+) total=(\d+)")

    with open(path) as f:
        for line in f:
            ts = parse_ts_us(line)

            m = epoch_phase.search(line)
            if m and ts is not None:
                ep = int(m.group(1))
                phase = m.group(2)
                dur = int(m.group(3))
                epochs[ep].main_phase_end_us[phase] = ts
                epochs[ep].main_phase_dur_us[phase] = dur
                if phase == "staging":
                    epochs[ep].pe_start_us = ts - dur
                if phase == "flush_start_async":
                    epochs[ep].flush_async_end_us = ts
                continue

            m = tl_line.search(line)
            if m:
                ep = int(m.group(1))
                offs: Dict[str, int] = {}
                for mm in re.finditer(r"(\w+)@(\d+)", m.group(2)):
                    offs[mm.group(1)] = int(mm.group(2))
                epochs[ep].tl_offsets = offs
                if "end" in offs:
                    epochs[ep].tl_total_us = offs["end"]
                continue

            m = worker_re.search(line)
            if m and ts is not None:
                workers.append(Worker(
                    end_ts_us=ts,
                    half1_us=int(m.group(1)),
                    sg_wait_us=int(m.group(2)),
                    half2_us=int(m.group(3)),
                    total_us=int(m.group(4)),
                ))
                continue

    return epochs, workers


# ---------- text helpers ----------

def _is_light(hex_color: str) -> bool:
    """Return True if the color is light enough that black text reads better."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    # Perceptual luminance (Rec. 601).
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


# ---------- rendering ----------

def render(target_epoch: int, epochs: Dict[int, EpochData],
           workers: List[Worker]) -> None:
    for ext in (".pdf", ".png"):
        p = OUT_BASE + ext
        if os.path.exists(p):
            os.remove(p)

    if target_epoch not in epochs:
        raise SystemExit(f"target epoch {target_epoch} not present in log")
    if target_epoch + 1 not in epochs:
        raise SystemExit(f"need epoch {target_epoch+1} for X-axis cropping")

    e      = epochs[target_epoch]
    e_next = epochs[target_epoch + 1]
    e_prev = epochs.get(target_epoch - 1)

    all_main_keys = (
        [k for _, k, _ in PRE_PE_PHASES]
        + ["staging"]
        + [k for _, k, _ in POST_PE_PHASES]
    )
    main_phase_starts = [
        e.main_phase_end_us[k] - e.main_phase_dur_us[k]
        for k in all_main_keys
        if k in e.main_phase_end_us
    ]
    if not main_phase_starts:
        raise SystemExit("no main-thread phases found for target epoch")
    t0 = min(main_phase_starts)

    next_main_starts = [
        e_next.main_phase_end_us[k] - e_next.main_phase_dur_us[k]
        for k in all_main_keys
        if k in e_next.main_phase_end_us
    ]
    next_epoch_us = min(next_main_starts) if next_main_starts else None
    if next_epoch_us is None:
        raise SystemExit("no main-thread phases found for next epoch")
    epoch_dur_us = next_epoch_us - t0
    x_max_us     = int(epoch_dur_us * (1.0 + EPOCHS_AFTER_TARGET_TO_SHOW))

    # Approximate axes width in inches after tight_layout; used by
    # add_bar_label to size labels so they fit their bar. 0.92 of fig width
    # is a reliable post-tight_layout estimate for our 15-inch wide plot.
    inches_per_ms = (15 * 0.92) / (x_max_us / 1000.0)

    def to_ms(us_abs: int) -> float:
        return (us_abs - t0) / 1000.0
    def width_ms(us: int) -> float:
        return us / 1000.0

    # Layout: 1 main row + 1 WB row.
    n_rows = 2
    row_pitch = 0.7
    fig_height = 1.9
    fig, ax = plt.subplots(figsize=(15, fig_height))

    row_h = 0.55
    sub_row_h = 0.45
    row_y: Dict[str, float] = {
        "main": 1 * row_pitch,
        "WB":   0 * row_pitch,
    }
    main_y = row_y["main"]

    # ---- Pre-PE phases ----
    for label, phase_key, color in PRE_PE_PHASES:
        if phase_key not in e.main_phase_end_us:
            continue
        end_us = e.main_phase_end_us[phase_key]
        dur_us = e.main_phase_dur_us[phase_key]
        start_us = end_us - dur_us
        if start_us < t0:
            continue
        x0 = to_ms(start_us); w = width_ms(dur_us)
        ax.barh(main_y, w, left=x0, height=row_h, color=color,
                edgecolor="black", linewidth=0.4)
        add_bar_label(ax, x0, w, main_y, label, dur_us, color, inches_per_ms)

    # ---- prepareEpoch sub-phases inline ----
    used_pe_subphases: List[str] = []
    pe_intervals: List[Tuple[int, int]] = []   # (start_us, end_us) of each colored PE block
    if e.pe_start_us is not None and e.tl_total_us is not None and e.tl_offsets:
        markers = sorted(e.tl_offsets.items(), key=lambda kv: kv[1])
        # The [TL] end marker can be a few us shorter than the outer
        # "Epoch N staging time" duration that initialization is anchored
        # against (small bookkeeping overhead between the inner end marker
        # and the outer log line). To avoid a visible gap between the last
        # PE sub-phase (remap) and initialization, extend the last colored
        # sub-phase to the outer staging end.
        outer_total_us = e.main_phase_dur_us.get("staging", e.tl_total_us)
        last_colored_idx = None
        for i, (name, _) in enumerate(markers[:-1]):
            if name not in PE_END_MARKERS and name in PE_SUBPHASE_CFG:
                last_colored_idx = i
        for i, (name, off) in enumerate(markers[:-1]):
            next_off = markers[i + 1][1]
            if name in PE_END_MARKERS:
                continue
            cfg = PE_SUBPHASE_CFG.get(name)
            if cfg is None:
                continue
            label, color = cfg
            if i == last_colored_idx and outer_total_us > next_off:
                next_off = outer_total_us
            seg_dur_us = next_off - off
            seg_start = e.pe_start_us + off
            seg_end   = seg_start + seg_dur_us
            if seg_start < t0:
                continue
            x0 = to_ms(seg_start); w = width_ms(seg_dur_us)
            ax.barh(main_y, w, left=x0, height=row_h, color=color,
                    edgecolor="black", linewidth=0.3)
            add_bar_label(ax, x0, w, main_y, label, seg_dur_us, color, inches_per_ms)
            if name not in used_pe_subphases:
                used_pe_subphases.append(name)
            pe_intervals.append((seg_start, seg_end))

    # ---- Post-PE phases ----
    for label, phase_key, color in POST_PE_PHASES:
        if phase_key not in e.main_phase_end_us:
            continue
        end_us = e.main_phase_end_us[phase_key]
        dur_us = e.main_phase_dur_us[phase_key]
        start_us = end_us - dur_us
        if start_us < t0:
            continue
        x0 = to_ms(start_us); w = width_ms(dur_us)
        ax.barh(main_y, w, left=x0, height=row_h, color=color,
                edgecolor="black", linewidth=0.4)
        add_bar_label(ax, x0, w, main_y, label, dur_us, color, inches_per_ms)

    # ---- WB row ----
    wb_active_intervals: List[Tuple[int, int]] = []   # h1 + h2 only (sg_wait excluded)

    def draw_wb(ep_data: EpochData, w_idx: int, h1_color: str, sw_color: str,
                h2_color: str, h1_label: str, sw_label: str, h2_label: str):
        if ep_data.flush_async_end_us is None:
            return
        if w_idx < 0 or w_idx >= len(workers):
            return
        worker = workers[w_idx]
        start_us = ep_data.flush_async_end_us
        h1_end = start_us + worker.half1_us
        sw_end = h1_end + worker.sg_wait_us
        h2_end = sw_end + worker.half2_us

        wb_active_intervals.append((start_us, h1_end))
        wb_active_intervals.append((sw_end, h2_end))

        for s_us, e_us, c, lab, dur in (
            (start_us, h1_end, h1_color, h1_label, worker.half1_us),
            (h1_end,   sw_end, sw_color, sw_label, worker.sg_wait_us),
            (sw_end,   h2_end, h2_color, h2_label, worker.half2_us),
        ):
            if e_us <= t0 or s_us >= t0 + x_max_us:
                continue
            vis_s = max(s_us, t0)
            vis_e = min(e_us, t0 + x_max_us)
            x0 = to_ms(vis_s); w = width_ms(vis_e - vis_s)
            ax.barh(row_y["WB"], w, left=x0, height=sub_row_h, color=c,
                    edgecolor="black", linewidth=0.3, zorder=2)
            # The label uses the un-clipped duration so the threshold
            # bucket reflects the true bar width, not the visible chunk.
            add_bar_label(ax, x0, w, row_y["WB"], lab, dur, c, inches_per_ms)

    if e_prev is not None:
        draw_wb(e_prev, target_epoch - 2,
                WB_HALF1_PREV, WB_WAIT_PREV, WB_HALF2_PREV,
                "half 1", "sync wait", "half 2")
    draw_wb(e, target_epoch - 1,
            WB_HALF1_THIS, WB_WAIT_THIS, WB_HALF2_THIS,
            "half 1", "sync wait", "half 2")

    # ---- Overlap shading: red alpha where active worker (h1/h2) work
    # happens concurrently with a prepareEpoch sub-phase. Drawn at zorder=0
    # so the colored bars stay solid on top; the shading shows in the gap
    # between the main and WB rows. Mirrors plotting/plot_overlap_timeline.py. ----
    for wb_s, wb_e in wb_active_intervals:
        for pe_s, pe_e in pe_intervals:
            os_v = max(wb_s, pe_s)
            oe_v = min(wb_e, pe_e)
            if oe_v > os_v:
                ax.axvspan(to_ms(os_v), to_ms(oe_v),
                           alpha=0.22, color="#E8B940", zorder=0)

    # ---- Axis cosmetics ----
    ax.set_xlim(0, x_max_us / 1000.0)
    ax.set_ylim(-row_pitch * 0.9, (n_rows - 1) * row_pitch + row_pitch * 0.9)
    ax.set_xlabel("ms (wall clock, t=0 at epoch start)")
    ax.set_yticks([row_y["main"], row_y["WB"]])
    ax.set_yticklabels(["main", "WB"], fontsize=12)
    ax.tick_params(axis="x", labelsize=12)

    # Vertical line at the next-epoch boundary; label inside the plot box.
    ax.axvline(epoch_dur_us / 1000.0, color="black", linestyle="--",
               linewidth=0.6, alpha=0.5)
    ax.text(epoch_dur_us / 1000.0, 0.985, " next epoch",
            fontsize=14, color="#444", va="top",
            transform=blended_transform_factory(ax.transData, ax.transAxes))

    # No legend: in-block labels make it redundant.

    # ax.set_title(
    #     f"YCSB-F rF/fT/120 B async epoch {target_epoch} (steady state)",
    #     fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight")
    fig.savefig(OUT_BASE + ".png", bbox_inches="tight", dpi=600)
    print(f"saved {OUT_BASE}.{{pdf,png}}")


def main() -> None:
    if not os.path.exists(LOG_PATH):
        raise SystemExit(f"log not found: {LOG_PATH}; run tests first")
    epochs, workers = parse_log(LOG_PATH)
    render(TARGET_EPOCH, epochs, workers)


if __name__ == "__main__":
    main()
