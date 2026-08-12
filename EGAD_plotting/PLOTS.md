# EGAD plot index

Reference for writing the paper. For each figure, this gives the headline
claim, the experimental design, what to point at when reading it, the
numbers worth citing, and what NOT to claim. Plot scripts in `plots/` have
the technical mechanism; this file has the paper-framing context.

Variance display (2026-08-11): every measured figure draws whiskers
spanning its repetitions (min to max), rendered from the same committed
log datasets. Center statistics were verified cell-exact against the
prior CSVs before the swap, so every cite table below still holds. The
CSVs gained min/max columns, and the rep-aggregation columns of figures
07/13 were renamed `mean_mtxn_s` to say what they always computed
(fig 09/15 genuinely use medians and keep `median_mtxn_s`). Figures 03
and 06 switched to median centers (`median_mtxn_s`) on 2026-08-12:
upstream's CPU executor shows a recurring slow mode on this hardware,
and the affected cells (every fig-03 cpu cell; fig 06 rTfF theta=0.8
and rFfT theta=0.6/0.8) carry ten reps, topped up on a quiet box, so
the median sits on the dominant mode instead of between modes.
Headline movements from that switch are disclosed in each figure's
section below.

---

## Hardware

- 2-socket Intel Xeon Silver 4410Y, 24 physical cores total (12 per socket, 24 logical via SMT per socket = 48 LCPUs)
- 384 GB total: 64 GB DDR5-4000 per CPU node plus a 256 GB CPU-less memory node (node 2, unused by figure runs)
- 3x NVIDIA RTX 5000 Ada (CC 8.9, 100 SMs, 32 GB GDDR6 per card), PCIe Gen4 x16
- NUMA: node 0 cores 0-11 (HT 24-35), node 1 cores 12-23 (HT 36-47); GPU 0 attached to node 0
- All measurements use GPU 0 only (`CUDA_VISIBLE_DEVICES=0`)

## Software

- Ubuntu 24.04, Linux 6.8, gcc 13, CUDA 12.x driver / NVCC 12.9
- CPU frequency governor set to `performance` before every benchmark run (default `schedutil` costs ~19% throughput at 120 B hybrid configs)
- Page cache dropped before every run (`sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`); already wired into every `tests/*.sh`

## Build

The record layout is a compile-time flag, so each record size needs two binaries.
ON (hybrid record layout) runs hybrid_staging and the EGAD modes. OFF (stock-Epic
layout) runs the cpu_only and gpu_only baselines: cpu_only is value-incorrect on ON,
and gpu_only's ON padding inflates its HBM footprint (stock EPIC-GPU runs well past
W=128, so its OOM ceiling on ON is an artifact, not Epic's real limit).

- `./build_binaries.sh --hybrid`                  -> `EGAD/build/epic_driver`            (ON, 1 KB YCSB + all TPCC)
- `./build_binaries.sh --hybrid --small-records`  -> `EGAD/build-small/epic_driver`      (ON, 120 B YCSB)
- `./build_binaries.sh`                           -> `EGAD/build-off/epic_driver`        (OFF, 1 KB YCSB + all TPCC baselines)
- `./build_binaries.sh --small-records`           -> `EGAD/build-small-off/epic_driver`  (OFF, 120 B YCSB baselines)

TPCC has fixed per-table record sizes by spec; hybrid_staging uses the ON binary
(`build`) while the cpu_only and gpu_only baselines use the OFF binary (`build-off`).

## YCSB run convention

Unless a plot notes otherwise, every YCSB hybrid_staging cell uses the
canonical headline config (`RUNNING_TESTS.md` §2):

```
EPIC_YCSB_CACHE_CAP=6720000  CUDA_VISIBLE_DEVICES=0
OMP_DYNAMIC=false OMP_NUM_THREADS=12
numactl --physcpubind=0-11,24-35 --membind=0
EGAD/build-small/epic_driver -b ycsb? -d epic -w 1 -a $skew
  -r true -c 32 -s 100000 -f false -m false -n 20000000
  -x gpu -e 300 -y hybrid_staging -z true
```

In words: 120 B records, -rT-fF, 20 M records, e=300, 100 K txns/epoch,
c=32 (24 for cpu_only), NUMA pin to node 0 with OMP_NUM_THREADS=12 for
hybrid only, `EPIC_YCSB_CACHE_CAP=6720000` (33.6 % cache/DB ratio) for
hybrid runs that need matched eviction pressure. Steady-state throughput
uses the **e250-e280** window (the long warmup is real; earlier windows
over-report).

### CPU baseline provenance (2026-07-22)

Every YCSB `cpu_only` cell in plots 03/05/06/07 (and the 20 M floor in
plot 13) runs an **unmodified upstream Epic build**
(ShujianQian/epic-artifact, epic @ 5a7dc90 = our fork base) with two
build-enabling changes only (the CC 8.9 CUDA-architecture entry and the
cuco Thrust guard) plus the record-size match for 120 B cells. Upstream
has no execution-mode flag, so its invocations drop `-y/-z`; everything
else is flag-identical. Two exceptions: plot 06's field-split configs
(-rT-fT / -rF-fT), where upstream's CPU mode refuses to run ("split
field not supported on CPU") and the baseline is our extended executor
(disclosed there), and all TPC-C cpu_only cells, which keep our fork's
binary, verified throughput-identical to upstream (0.99x to 1.00x on
tpcc and tpccfull at W=32/128). Campaign logs, interleaved same-day
controls, and the upstream Figure 7/8 replications live in
`verify_runs/stock_cpu_2026-07-17/` (NOTES.md there is the record).
The prior our-executor cpu logs are archived per convention as
`logs/<name>.pre_stockcpu_20260722/`.

## TPCC run convention

TPCC hybrid_staging cells use the canonical TPCC config
(`RUNNING_TESTS.md` §2):

```
EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1
EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1
EPIC_PREPARE_EPOCH_PARALLEL=1 EPIC_WORKLOAD_AWARE_AUTOSIZER=1
OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false
CUDA_VISIBLE_DEVICES=0
numactl --physcpubind=0-11,24-35 --membind=0
EGAD/build/epic_driver -b tpccdeck -d epic -w 128 -a 0.0
  -r false -c 24 -s 100000 -f false -m false -n 20000000
  -x gpu -e 50 -y hybrid_staging -z true
```

Default mix is `tpccdeck` (NewOrder/Payment/OrderStatus/Delivery/StockLevel
= 10/10/1/1/1, OL backlog bounded by NO=Delivery balance). The spec mix
`tpccfull` (45/43/4/4/4) appears only in plot 11's cliff demo. Steady-state
window is **e30-e50**.

cpu_only TPCC uses the canonical as-shipped invocation (`RUNNING_TESTS.md`
§2): no env vars, no NUMA pin,
`-c 24` (`CpuExecutor` spawns -c std::threads per epoch; spawn cost ceilings
beyond 24 cores). gpu_only TPCC uses the EGAD flat-index env vars
(`EPIC_FLAT_AUX_INDEX=1 EPIC_FLAT_INDEX_OL=1 EPIC_MIX_REALISTIC_SIZING=1`)
for fairness with hybrid, no NUMA, no staging-specific env vars. Both baselines
run on the OFF binary (`build-off`); only hybrid_staging uses the ON binary.

## Replication

Each plot has a `tests/NN_*.sh` that produces the source logs under
`logs/<plot_name>/` and a `plots/NN_*.py` that renders the figure from
those logs. `run_all.sh` orchestrates the full pipeline in lex order.
Raw logs are gitignored (regenerable from `tests/`); per-plot CSVs in
`figures/<plot_name>.csv` preserve the cell-level medians for re-styling
without re-running.

---

## 01 - TPCC overlap timeline (pre-existing)

`figures/tpcc_overlap.{pdf,png}` from `plots/01_tpcc_overlap.py`

### Headline
A single steady-state TPC-C epoch rendered as a Gantt chart showing the
eight per-table stagers' main-thread phases plus the concurrent async
writeback workers from the previous epoch.

### Experimental design
TPC-C, hybrid_staging, async writeback, W=128, deck mix, e=50, epoch-46
target (inside the e30-50 steady-state window). The figure shows one
mid-run epoch as a stacked timeline.

### How to read
- Main-thread bars: each stager's prepareEpoch sub-phases (sg_transfer
  dominates) plus init/exec/flush_start_async.
- Async worker bars: previous epoch's writeback that runs concurrently.
- Overlap shading shows where worker D2H/scatter coexists with main-thread
  staging.

### Cite
- The overlap density between worker activity and main-thread staging.
- order_line is the critical-path writeback (its bar extends longest).

### Caveats
- Single epoch; per-epoch shape varies. The overlap claim is about the
  steady-state pattern, not this specific epoch.

---

## 02 - YCSB-F overlap timeline (pre-existing)

`figures/ycsb_overlap.{pdf,png}` from `plots/02_ycsb_overlap.py`

### Headline
Single steady-state YCSB-F epoch with the prepareEpoch sub-phases
expanded inline and the async writeback worker shown as half1/sg_wait/half2.

### Experimental design
ycsbf, 120 B, -rF-fT, hybrid_staging async, e=300, e260 target.

### How to read
- Main row: idx_tsf, indexing, submit, then prepareEpoch sub-phases
  (build needed, fifo select/evict, sg_transfer, remap), then init, exec,
  flush_sync_prev, flush_start_async.
- WB row: this epoch's worker (red) and previous epoch's worker (pale red)
  with a sync_wait gap between half1 and half2.
- Gold overlap shading marks where workers run concurrently with the
  main-thread prepareEpoch sub-phases.

### Cite
- The async writeback fully overlaps with the next epoch's prepareEpoch.
- sg_wait is the worker's intentional yield (not real work).

### Caveats
- Single epoch.
- This is at -rF-fT, not the headline -rT-fF config. The phase shapes are
  similar but absolute values differ from plots 03-08.

---

## 03 - Record size sensitivity

`figures/record_size.{pdf,png,csv}` from `plots/03_record_size.py`

### Headline claim
"At matched cache pressure, 120 B records deliver 4.2 to 6.2 times higher
hybrid throughput than 1 KB records below theta=0.99 (per-byte PCIe
cost), and 1 KB hybrid sits below 1 KB cpu_only for theta<=0.6 (the
staging bandwidth ceiling), crossing above at theta=0.8."

### Experimental design
- ycsbf, 120 B vs 1 KB, hybrid + cpu_only
- cpu_only = unmodified upstream Epic (see CPU baseline provenance above);
  ALL cpu cells are medians of 10 reps as of 2026-08-12 (120 B cells
  topped up from 3 on the quiet box; recurring slow mode, see caveats);
  hybrid cells are 3-rep medians (tight)
- 20 M records, 6 skews (0.01, 0.2, 0.4, 0.6, 0.8, 0.99)
- 1 KB hybrid uses autosizer's natural ~10.54 M record cache (~52.7 %
  ratio)
- 120 B hybrid uses `EPIC_YCSB_CACHE_CAP=10544194` to match the same
  ~52.7 % ratio (its natural autosize would be 20 M = full DB)
- The cap value matches plot 03's earlier convention (slightly different
  from plots 05-08's 33.6 % cap; see Appendix B)

### How to read
- Solid blue (120 B hybrid) is the headline curve, above its baseline
  from theta=0.2 up.
- Solid red (1 KB hybrid) sits BELOW dashed red (1 KB cpu_only) through
  theta=0.6 - the "staging bandwidth ceiling" region - and pulls above
  from theta=0.8.

### Cite
- 120 B: hybrid wins from theta=0.2 up (1.14x to 1.98x; peak 1.98x at
  theta=0.8, 15.13 vs 7.63 MTxn/s). theta=0.01 is just below (0.95x,
  8.69 vs 9.13). theta=0.99 is 1.28x (5.00 vs 3.91; contention-bound).
- 1 KB: hybrid below cpu_only at theta<=0.6 (0.61x at theta=0.01, 1.54 vs
  2.53; 0.94x at theta=0.6), then 1.28x at theta=0.8 and 1.33x at
  theta=0.99 (3.80 vs 2.86).
- 120 B / 1 KB hybrid ratio: 4.2x to 6.2x below theta=0.99; 1.3x at
  theta=0.99 where GPU-executor contention bounds both record sizes.
- (Medians of ten resist the cpu slow mode; the old 3-rep means sat
  between modes, which is why several ratios moved with the 2026-08-12
  top-up: peak 2.12x->1.98x, 1 KB high-skew 1.76x->1.33x.)

### Caveats
- This is an apples-to-apples per-byte cost comparison, NOT a
  recommendation to deploy at 1 KB. Plot 07's cache sweep is what shows
  the design's response to cache pressure.
- Uses 52.7 % cache cap (not the 33.6 % used in plots 05-08); see
  Appendix B.
- Upstream's CPU executor is bimodal at 1 KB on this box: a slow mode
  (up to ~100x slower epochs at high skew, entirely in the execution
  phase, sometimes flipping mid-run) recurs in 2-4 of 10 reps per cell.
  The plotted values are medians of 10 reps at the standard (2,5)
  window; per-rep distributions are in
  verify_runs/stock_cpu_2026-07-17/NOTES.md. Our fork's executor shows
  no such mode (its 25dac65 rewrite), which is why the pre-swap cpu
  line looked flatter.

### Related
- Plot 05 (cross-workload validation at headline)
- Plot 07 (cache sensitivity at fixed record size)

---

## 04 - Staging overhead when not needed

`figures/three_way.{pdf,png,csv}` from `plots/04_three_way.py`

### Headline claim
"When the database fits in HBM and gpu_only runs natively, the
hybrid_staging machinery costs 9 % to 48 % of the gpu_only throughput
ceiling, depending on workload write fraction."

### Experimental design
- 4 workloads (ycsba, ycsbb, ycsbc, ycsbf)
- 120 B, 5 M records (small enough that DB fits HBM)
- 6 skews
- 2 modes: gpu_only and hybrid_staging
- cpu_only is intentionally excluded - at this DB size no one would pick
  cpu_only over gpu_only

### How to read
- Solid blue = hybrid; dashed light blue = gpu_only ceiling.
- The gap between them at each skew is the staging machinery's overhead.

### Cite
- ycsbc (read-only): hybrid hits 91 % of gpu_only ceiling - minimal
  overhead since writeback is near-zero.
- ycsbb (95R/5U): hybrid hits 76 to 85 % across skews.
- ycsba and ycsbf (50R/50U or RMW): hybrid hits 52 % at low skew (lots
  of unique modified records, big writeback work), recovers to 96 % at
  theta=0.99 (writeback concentrates on hot records, less per-record cost).

### Caveats
- This is the "what does staging cost when not strictly needed" plot.
  It is NOT a contribution plot. Contribution plots are 05 and 07.
- At theta=0.99 the gpu_only ceiling itself collapses for ycsba/f
  (GPU-side hot-record contention) - hybrid matches because both share
  the same bottleneck.

### Related
- Plot 05 (the "when staging IS needed" contribution plot)
- Plot 07 (cache sweep showing where staging becomes worth its overhead)

---

## 05 - Cross-workload headline (hybrid vs cpu_only)

`figures/beyond_hbm.{pdf,png,csv}` from `plots/05_beyond_hbm.py`

### Headline claim
"At the headline configuration with matched 33.6 % cache pressure, the
hybrid-vs-cpu comparison splits by contention regime. cpu_only
(unmodified upstream Epic) holds the low-skew cells (9 of the 16
theta<=0.6 cells sit at or below parity, 0.71x to 1.10x), and
hybrid_staging wins every theta>=0.8 cell, up to 3.10x (ycsbb at
theta=0.99)."

### Experimental design
- 4 workloads (ycsba, ycsbb, ycsbc, ycsbf)
- 120 B, -rT-fF, 20 M records
- 2 modes: hybrid (with `EPIC_YCSB_CACHE_CAP=6720000` = 33.6 % ratio)
  and cpu_only
- 6 skews
- gpu_only is absent: at this matched cache pressure the "would you use
  gpu_only" question is irrelevant; this plot is the hybrid-vs-cpu story
  specifically

### How to read
- 4 panels, one per workload, with solid blue = hybrid and dashed light
  blue = cpu_only.
- The dashed cpu line starts above or level with the solid hybrid line
  at low skew and falls below it as contention rises; the crossover
  sits between theta=0.4 and theta=0.8 depending on workload.

### Cite
- theta=0.01 (low skew, where staging is hardest and upstream's CPU
  executor is fastest): ycsba 1.00x, ycsbb 0.78x, ycsbc 0.84x,
  ycsbf 0.71x.
- theta=0.8: ycsba 1.34x, ycsbb 2.14x, ycsbc 1.95x, ycsbf 1.57x.
- theta=0.99: ycsbb 3.10x, ycsbc 3.02x (40.12 MTxn/s absolute),
  ycsba 1.29x, ycsbf 1.30x. The write-heavy pair drops in absolute
  terms there (GPU-executor hot-record contention, same as gpu_only)
  but stays above cpu_only because upstream's CPU executor collapses
  harder under contended writes (3.9 MTxn/s).

### Caveats
- The 1 KB story is in plot 03, not here. Plot 05 was originally scoped
  for both record sizes but trimmed to 120 B only after the record-size
  decision landed.
- Hybrid cells re-measured 2026-06-12 at the correct 33.6 % cap on the
  pinned epic SHA (6c52f8b); the 2026-05-14 originals had a 52.7 % cap
  copy-pasted from plot 03 (preserved in logs/beyond_hbm_52pct_oldbuild/
  and figures/beyond_hbm_52pct.csv.bak).
- cpu_only cells re-measured 2026-07-17 on unmodified upstream Epic
  (see CPU baseline provenance); interleaved same-day controls
  reproduced the prior our-executor numbers, whose logs are archived in
  logs/beyond_hbm.pre_stockcpu_20260722/. The ycsbb theta=0.01 stock
  cell carries 10 reps (both executors are bistable there;
  verify_runs/stock_cpu_2026-07-17/NOTES.md).

### Related
- Plot 03 (record size sensitivity at single workload)
- Plot 04 (the gpu_only ceiling comparison)

---

## 06 - Configuration sensitivity (where the design falls short)

`figures/config_sensitivity.{pdf,png,csv}` from `plots/06_config_sensitivity.py`

### Headline claim
"The hybrid design's win over cpu_only depends on read/write semantics.
At -rT-fT (full-record reads against field-split storage), the 10x
read-op amplification eats throughput and hybrid loses to cpu_only at
every skew except a tie at theta=0.8."

### Experimental design
- ycsbf, 120 B, 20 M records
- 3 configs: -rT-fF (headline), -rT-fT (full read + field split,
  worst case), -rF-fT (single field read + field split, best case)
- 2 modes per config: hybrid and cpu_only
- cpu_only provenance differs by config: -rT-fF runs unmodified upstream
  Epic; -rT-fT and -rF-fT run our extended executor, because upstream's
  CPU mode refuses field-split configs outright ("split field not
  supported on CPU", verified 2026-07-17). The 25dac65 rewrite added
  that capability; the paper discloses it where the figure is discussed.
- Centers are per-cell medians (2026-08-12). The slow-mode cpu cells
  (rTfF theta=0.8: reps down to 1.66; rFfT theta=0.6/0.8: reps down to
  5.25/3.08) carry ten reps each; all other cells are 3-rep medians
  (tight). Movements vs the old 3-rep means: rTfF theta=0.8 ratio
  2.52x -> 1.53x (the old cpu mean sat in the slow mode), rTfF
  theta=0.99 1.19x -> 1.28x, low-skew band 0.91-0.97x -> 0.90-0.93x,
  rFfT theta=0.8 2.08x -> 1.85x.
- Cache pressure held at ~33.6 % across configs:
  - -rT-fF: `EPIC_YCSB_CACHE_CAP=6720000` (records, 33.6 % of 20 M)
  - -rT-fT and -rF-fT: no cap (autosizer's natural pick = 67 M field-units,
    which is exactly 33.6 % of 200 M field-units)
- 6 skews

### How to read
- 6 lines, 3 colors x {hybrid solid, cpu_only dashed}:
  - Blue (-rT-fF): regime split, hybrid 0.90x to 0.93x at theta <= 0.4,
    then 1.21x at 0.6, 1.53x at 0.8, 1.28x at 0.99
  - Red (-rT-fT): hybrid LOSES to cpu_only at every skew except a
    1.01x tie at theta=0.8 (0.55 to 0.81x elsewhere). This is the
    "falls short" line.
  - Green (-rF-fT): hybrid at or above parity everywhere, biggest at
    theta=0.99 (1.88x) and theta=0.8 (1.85x)

### Cite
- At -rT-fT, hybrid throughput is 1.9 to 3.5 MTxn/s vs cpu_only 3.0 to
  3.8 MTxn/s - at or below baseline across the sweep.
- At -rT-fF, hybrid 5.0 to 11.5 MTxn/s vs stock cpu_only 3.9 to 9.0.
- At -rF-fT, hybrid up to 17.3 MTxn/s at theta=0.99, the absolute peak
  (1.88x its cpu baseline there).

### Caveats
- ycsbf only. Cross-workload would multiply by 4x but the per-config
  ordering is the same across workloads (verified in plot 05 for -rT-fF).

### Related
- Plot 03 (record size at the -rT-fF headline)
- Plot 04 (staging overhead at -rT-fF)
- Plot 05 (cross-workload at -rT-fF)

---

## 07 - Cache size sensitivity (the elbow)

`figures/cache_sensitivity.{pdf,png,csv}` from `plots/07_cache_sensitivity.py`

### Headline claim
"hybrid_staging throughput grows with cache size, reaching ~15.3 MTxn/s
at 75 % cache (1.79x the cpu_only floor, 44 % of gpu_only's ceiling).
Against the stock cpu_only floor of 8.59, the win threshold at
theta=0.5 sits between 25 % and 50 % cache; below it the CPU baseline
is faster."

### Experimental design
- ycsbf, 120 B, 20 M records, theta=0.5
- Cache cap sweep: 5, 10, 25, 50, 75, 100 % of DB
- gpu_only and cpu_only flat reference lines

### How to read
- Solid blue curve (hybrid_staging) bends down at low cache ratios.
- Green dashed (gpu_only ceiling, 34.6 MTxn/s).
- Red dashed (cpu_only floor, 8.59 MTxn/s, unmodified upstream Epic;
  see CPU baseline provenance).
- The hybrid curve crosses the cpu floor between 25 % and 50 % cache,
  then climbs to its plateau region at 75 %.

### Cite
- Hybrid at 5 % cache: 5.44 MTxn/s (0.63x the cpu floor).
- Hybrid at 25 % cache: 7.01 MTxn/s (0.82x, still under the floor).
- Hybrid at 50 % cache: 11.97 MTxn/s (1.39x).
- Hybrid at 75 % cache: 15.34 MTxn/s (1.79x, 44 % of the gpu ceiling);
  100 % measures 14.04, within the plateau's run-to-run band.
- Deployment guidance at theta=0.5: the cache must hold roughly half
  the records to beat the CPU fallback, 75 % for the plateau. At higher
  skew the thresholds shrink (the hot set fits a smaller cache).

### Caveats
- Single workload (ycsbf). Other workloads may have different elbow
  shapes (ycsbc would be flatter since reads scale with skew, not cache).
- Single skew (theta=0.5). At very high skew the curve flattens earlier
  (hot set fits a smaller cache).

### Related
- Plot 04 (overhead at 100 % cache equivalent)
- Plot 05 (matched at 33.6 % ratio across workloads)

---

## 08 - Async vs sync writeback (per-phase breakdown)

`figures/writeback_breakdown.{pdf,png,csv}` from `plots/08_writeback_breakdown.py`

### Headline claim
"Async writeback removes the synchronous-flush bottleneck from per-epoch
latency. On write-heavy workloads (ycsba, ycsbf) it cuts epoch time by
~18 to 20 % by hiding ~3.5 ms of writeback work behind the next epoch's
main-thread phases."

### Experimental design
- 4 workloads, 120 B, 20 M records, theta=0.5
- 33.6 % cache cap (matched-ratio convention)
- 2 modes: hybrid_staging with -z true (async) vs -z false (sync)
- 3 reps, e=40 (steady-state window e30-e40)
- 8 stacked horizontal bars (4 workloads x 2 modes) with 8 phase blocks:
  index_transfer, indexing, submission, transfer (compute), transfer (data),
  initialization, execution, writeback

### How to read
- The fat red "writeback" block on sync bars (ycsba and ycsbf) is what
  async hides.
- transfer (data) is the dominant phase (~80 % of staging) - the actual
  PCIe + scatter.
- transfer (compute) is the staging management work (build needed set,
  FIFO eviction, remap).
- writeback in async mode is flush_sync_prev + flush_start_async only;
  the rest of the writeback runs in parallel on the worker thread and is
  invisible to the main thread.

### Cite
- ycsba/ycsbf: sync 12.5 ms epoch vs async 10.2 ms (~18 to 20 % savings).
- Sync writeback block: ~4.3 ms (35 % of sync epoch).
- Async writeback block: ~0.8 ms (8 % of async epoch).
- transfer (data) is ~80 % of staging across all workloads.
- ycsbc (read-only): async = sync (~7.6 ms each) - no writeback to hide.
- ycsbb (95R/5U): nearly tied (8.4 ms vs 8.3 ms) - small writeback volume.

### Caveats
- Single skew (theta=0.5). At higher skew the writeback set is smaller
  so the async win shrinks.
- In async mode, `index_transfer` is ~3.6x larger than sync (954 us vs
  263 us on write-heavy workloads). This is host-side runtime contention
  between the worker thread and main-thread CUDA submission, not a PCIe
  effect. See Appendix C.

### Related
- Plot 02 (single-epoch timeline showing the overlap visually)

---

## 09 - TPCC warehouse-count sweep (headline scaling, two mixes)

`figures/tpcc_warehouse_sweep.{pdf,png,csv}` from `plots/09_tpcc_warehouse_sweep.py`

### Headline claim
"Across TPC-C tpccdeck (deck mix) AND TPC-C NP (NewOrder + Payment
only — Epic's `tpcc` benchmark), hybrid_staging beats cpu_only at
every swept W from 8 up. gpu_only traces the in-HBM ceiling on both
panels; on NP its device footprint (tables + insert pools provisioned
for the full run) exhausts the 32 GB card above W=208, and
hybrid_staging continues past it through W=240 at 1.26-1.32x over
cpu_only. tpccdeck is swept to W=128, where hybrid_staging delivers
1.41x over cpu_only. The two mixes are shown side by side: the NP
panel matches Epic's Figure 5 conventions, and tpccdeck is EGAD's
canonical deep-run mix."

### Experimental design
- **1x2 panel layout**:
  - Left: `tpccdeck` (deck mix 10/10/1/1/1, OL backlog bounded by
    NewOrder/Delivery balance — EGAD's canonical deep-run mix)
  - Right: `tpcc` (NP — 50/50/0/0/0 = NewOrder + Payment only, the
    Epic-paper benchmark variant used in Epic's Figure 5)
- Both panels share the Y axis (sharey=True) so cross-mix comparisons
  read directly.
- W ∈ {4, 8, 16, 32, 64, 128} on tpccdeck; NP extends the sweep with
  W ∈ {160, 192, 208, 216, 240}. 3 reps each, e=50 (steady-state
  window e30-e50).
- Three modes per panel:
  - **hybrid_staging**: canonical `RUNNING_TESTS.md` §2 env-var set (flat
    indexes, prewarm, parallel sections, workload-aware autosizer,
    NUMA pin, OMP 8,3).
  - **cpu_only**: canonical `RUNNING_TESTS.md` §2 invocation (no env vars, no NUMA,
    `-c 24`).
  - **gpu_only**: stock OFF-build gpu_only (no NUMA, no
    staging-specific env vars). On NP its device footprint exceeds
    the card above W=208, so its line ends there and the plot drops
    the missing cells; on tpccdeck it runs at every swept W.

### How to read
- X-axis (each panel) is W (log-2). Y-axis is steady-state throughput
  (MTxn/s), shared across panels for direct cross-mix comparison.
- Solid blue (hybrid_staging) sits above dashed red (cpu_only) at
  every swept W from 8 up on both panels; at W=4 it sits at or below
  cpu_only.
- Dashed green (gpu_only) traces the in-HBM ceiling; on NP it ends
  above W=208 where the device footprint exhausts the card, while
  hybrid_staging and cpu_only continue to W=240.
- The NP panel sits visibly higher than the tpccdeck panel because
  NP has no Delivery / OrderStatus / StockLevel scans; matches
  Epic's Figure 5 observation that NP throughput exceeds full-mix.

### Cite (measured 2026-07-06, epic@996e4a5, e30-e50 median across 3 reps)

| tpccdeck  | W=4  | W=8  | W=16 | W=32 | W=64 | W=128 |
|-----------|-----:|-----:|-----:|-----:|-----:|------:|
| hybrid_staging | 5.50 | 7.39 | 8.72 | 9.21 | 8.87 | **8.56** |
| cpu_only       | 5.75 | 5.96 | 6.16 | 6.14 | 6.25 | 6.08     |
| gpu_only       | 7.48 | 11.65| 16.11| 17.62| 18.42| 18.63    |

| tpcc (NP) | W=4  | W=8  | W=16 | W=32 | W=64 | W=128 | W=160 | W=192 | W=208 | W=216 | W=240 |
|-----------|-----:|-----:|-----:|-----:|-----:|------:|------:|------:|------:|------:|------:|
| hybrid_staging | 6.20 | 9.17 | 12.11| 12.65| 12.30| **11.17** | 11.16 | 11.03 | 10.52 | 10.61 | 10.28 |
| cpu_only       | 7.41 | 8.03 | 7.96 | 8.55 | 8.59 | 8.21  | 8.76  | 8.51  | 8.97  | 8.42  | 7.76  |
| gpu_only       | 7.38 | 12.17| 18.32| 22.51| 25.57| 27.44 | 28.48 | 28.27 | 28.31 | —     | —     |

- **W=128 headline**: tpccdeck hybrid 8.56 vs cpu_only 6.08 (1.41x);
  NP 11.17 vs 8.21 (1.36x).
- **Past the NP gpu_only ceiling** (device footprint exceeds the card
  above W=208): hybrid 10.61 vs cpu_only 8.42 at W=216 (1.26x) and
  10.28 vs 7.76 at W=240 (1.32x). This is the regime the staging
  design exists for.
- **NP gpu_only ceiling is dramatically higher** than tpccdeck's
  (W=64: 25.57 vs 18.42, +39 %). The deck mix's Delivery /
  OrderStatus / StockLevel txns add OL scanning work that shows up
  in the execution kernel with no staging involved.
- cpu_only is roughly flat across W on both mixes (per-txn work
  depends on the mix, not W): tpccdeck 5.75-6.25, NP 7.41-8.97.
- At W=4 the staging machinery is overkill (DB fits HBM trivially):
  hybrid lands at 0.96x cpu_only on tpccdeck and 0.84x on NP, its
  only swept point at or under the CPU baseline. Hybrid wins from
  W=8 up (tpccdeck 1.24x, NP 1.14x).

### Caveats
- **W=1 and W=2 are excluded.** The underlying Epic TPCC pipeline
  exhibits intermittent (~20-50%) GPU-side memory-access instability
  at very low warehouse counts that affects ALL three execution modes
  (cpu_only too, since cpu_only still runs the GPU indexer per Epic's
  design). The instability pre-dates this codebase; the current build
  ships 11 defensive correctness fixes that partially mitigate it but
  do not eliminate the underlying race in the indexer/executor
  pipeline. W ≥ 4 is rock-solid across all modes.
- The Epic-Figure-8 "CPU wins at W=1 due to GPU contention" observation
  is therefore not measurable on this codebase; cite Epic's number
  from the literature rather than attempting to re-measure.
- The NP W=128 hybrid cell has a ~±4 % run-to-run band; the table
  quotes the campaign median (11.17). The neighboring W=160 cell
  (11.16) corroborates the level.

### Related
- Plot 01 (per-stager overlap timeline at W=128)
- Plot 10 (per-phase breakdown at W=128, async vs sync)
- Plot 11 (long-run cliff at W=128 under tight HBM)

---

## 10 - TPCC async vs sync writeback (per-phase breakdown)

`figures/tpcc_writeback_breakdown.{pdf,png,csv}` from `plots/10_tpcc_writeback_breakdown.py`

### Headline claim
"At W=128 tpccdeck, async writeback hides the per-table flush across
all 8 stagers behind the next epoch's prepareEpoch, and its ordered
scatter also finishes the same writeback faster than sync mode's
cache-order walk. Steady-state (e30-e50) epoch wallclock drops from
25.56 ms (sync) to 11.26 ms (async), a 2.27x reduction. The writeback
block on the main thread's critical path shrinks from 16.81 ms (66%
of the sync epoch) to 1.79 ms (16% of the async epoch), a 9.4x
reduction."

### Experimental design
- Single workload (tpccdeck), W=128, e=50, 3 reps.
- Two modes: async (-z true) vs sync (-z false). Both run with the
  same canonical `RUNNING_TESTS.md` §2 env-var set.
- Per-epoch phases (e30-e50 median): index_transfer, indexing,
  submission, prepareEpoch (the 8-stager parallel block, wallclock =
  max across stagers since they overlap), initialization, execution,
  writeback (= flush in sync mode, = flush_sync_prev + flush_start_async
  in async mode; the real D2H/scatter runs on per-stager workers and is
  invisible on the main-thread bar).

### How to read
- Two horizontal stacked bars stacked vertically: top = async, bottom = sync.
- The fat red "writeback" block on the sync bar is what async hides.
- prepareEpoch dominates both bars; index_transfer + indexing + submission
  are similar across modes.

### Cite (measured 2026-07-06, epic@996e4a5, e30-e50 median across 3 reps)
- Async total wallclock: 11.26 ms; sync total wallclock: 25.56 ms.
  Async is 2.27x faster on epoch wallclock.
- Writeback block: sync 16.81 ms (66% of epoch) shrinks to async
  1.79 ms (16% of epoch), a 9.4x reduction.
- prepareEpoch is the dominant non-writeback block in both modes
  (3.12 ms sync, 3.26 ms async, +5% in async from the cross-NUMA
  contention with the concurrently-running writeback worker, see
  Appendix C for the YCSB analog of this effect).
- indexing, submission, init, and exec are within ~10% across modes;
  index_transfer roughly doubles in async (505 us -> 1064 us), the
  CUDA-runtime contention tax also documented in Appendix C.

### Caveats
- prepareEpoch is rendered as a single block aggregating all 8
  stagers' wallclocks. The per-stager breakdown lives in plot 01's
  Gantt timeline.
- Single workload (tpccdeck). The async/sync ratio depends on the
  ratio of OL eviction volume to per-epoch wallclock; tpccfull would
  show a similar shape pre-cliff (see plot 11).

### Related
- Plot 08 (YCSB analog at -rT-fF 120 B)
- Plot 01 (per-stager overlap timeline at W=128)

---

## 11 - TPCC deep-E cliff (where the design falls short)

`figures/tpcc_cliff.{pdf,png,csv}` from `plots/11_tpcc_cliff.py`

### Headline claim
"At W=128 with --hybrid_hbm_reserve_gb=6 (deliberately undersized
cache), tpccdeck holds a 7.0-8.9 MTxn/s band through e=390 with no
downward trend because its NewOrder/Delivery balance keeps the OL
backlog bounded at ~9000W. The tpccfull spec mix peaks near 9.0
by e=50-100, then declines monotonically to ~5.5 by e=350 as the
unbounded undelivered queue overflows the cache and eviction starts
hitting still-needed slots. tpccfull crosses under cpu_only's 6.01
floor by ~e=316, eliminating the hybrid advantage."

### Experimental design
- W=128, --hybrid_hbm_reserve_gb=6 (cache pressure deliberately tight).
- Two mixes side-by-side: tpccdeck (bounded queue) vs tpccfull (spec
  mix, unbounded undelivered queue).
- 3 reps each, e=400 (long enough to expose the cliff dynamic).
- Plus one cpu_only baseline cell at W=128 tpccdeck for the floor
  reference line (cpu_only is flush-free so cache pressure does not
  affect it).
- `EPIC_NO_HUGE_GROWING=1`: this deep-run demo isolates the GPU-cache
  capacity dynamic, so the growing-store huge-page allocation policy
  is held at 4 KB. With the madvise active, tpccfull's decline onset
  moves from ~e=300 to ~e=120 (reproduced under defrag=defer and with
  numa_balancing off — a host-allocation effect orthogonal to the
  cache-pressure story). The e<=50 figures keep the madvise (default).
  Deep runs also require THP defrag=defer (test/preflight auto-set).

### How to read
- X-axis is epoch number 1-400. Y-axis is per-epoch instantaneous
  throughput (NOT a steady-state window).
- Solid blue (tpccdeck): flat across the run, demonstrates the design
  is sustainable when working set is bounded by construction.
- Solid red (tpccfull): climbs to peak by ~e30, holds through ~e100,
  then declines monotonically as eviction misses pile up.
- Dashed gray (cpu_only floor): horizontal line; tpccfull cliffs
  through it by ~e=316 and continues below.

### Cite (measured 2026-07-07, epic@996e4a5, 3 reps, e=400)
- **tpccdeck holds a 7.0-8.9 MTxn/s band through e=390** (min 7.01,
  max 8.86 across e=50-390, no downward trend), confirming the deck
  mix's bounded OL backlog sustains the hybrid_staging advantage
  across the run.
- **tpccfull peaks at ~8.95 MTxn/s by e=52 (8.60 at e=100, 7.71 at
  e=120), declines to 6.90 by e=250 and ~5.5 by e=350**, sustained
  under cpu_only's floor (6.01) from ~e=316. The decline is monotonic
  from ~e=100 onward, not a sharp cliff.
- Hybrid advantage over cpu_only: +43% pre-cliff (e=100, 8.60 vs
  6.01), -8% post-cliff (e=350, 5.53). The OL undelivered queue
  overflows the OL cache inside the [e=100, e=350] window under
  `--hybrid_hbm_reserve_gb=6`.
- cpu_only floor: 6.01 MTxn/s (measured at e=50 since cpu_only at
  e=400 OOMs the GPU on the OL primary-store insert-pool reservation
  scaling with epochs; cpu_only's throughput is constant with W and
  E for steady-state purposes, so e=50 is sufficient).

### Caveats
- Cliff onset epoch is sensitive to --hybrid_hbm_reserve_gb (we use
  6 GB to make the cliff appear within e=400; at the default 4 GB
  reserve the onset sits past e=295).
- tpccfull's monotonic-growth pathology is a property of the spec
  mix's NewOrder/Delivery imbalance, not a property of EGAD's design;
  see related work for systems that handle unbounded growth via
  partitioning.

### Related
- Plot 09 (no-cliff scaling at default cache budget)

---

## 12 - YCSB cache-warmup trajectory (1 KB records, all 6 skews)

`figures/ycsb_warmup.{pdf,png,csv}` from `plots/12_ycsb_warmup.py`

### Headline claim
"At 1 KB records, every skew starts cold at low throughput (every
read is a miss-admit and the stager ships 8x more bytes per record
than at 120 B). Over the next ~10-40 epochs each line climbs as the
working set admits into cache, until per-epoch admit volume exceeds
free cache slots and FIFO eviction starts. The eviction-onset epoch
is marked per skew with a thin red dashed vertical line, after which
the curve plateaus at its post-eviction steady state. Low-skew runs
trigger eviction within the first ~20 epochs (wide working set fills
cache fast). High-skew runs (theta>=0.8) never trigger eviction
because the hot set fits entirely in cache."

### Experimental design
- ycsbf, **1 KB records**, -rT-fF, 20 M records, autosizer's natural
  cache pick (~52.7 % cache/DB ratio at this DB size).
- 6 skews: θ ∈ {0.01, 0.2, 0.4, 0.6, 0.8, 0.99} (standard sweep).
- 3 reps per skew, e=300.
- **Data is shared with plot 03's `record_size/` 1KB hybrid_staging
  cells** to avoid re-running. The plot 12 test script verifies the
  source logs exist and references them.
- Per-epoch throughput = `100000 / (t_{e+1} - t_e)` from `Running
  epoch N` timestamps.
- Eviction-onset per skew = first log epoch where the stager emits
  `Evictions needed:` (i.e., per-epoch admit set first exceeds the
  free-list ring's free slots), median across 3 reps.

### How to read
- X-axis: epoch number 1-300. Y-axis: per-epoch throughput (MTxn/s).
- One line per skew, colored by viridis ramp (low-skew=purple,
  high-skew=yellow).
- **Thin red dashed vertical line** = the median epoch where that
  skew first triggered FIFO eviction. Lines are present only for
  skews that ever evict; theta=0.8 and 0.99 have no vertical
  marker because their hot set never overflows the cache.
- Each curve's slope flattens around its red line (warmup -> plateau
  transition). The plateau height after the line is the
  steady-state with continuous eviction.

### Cite (measured 2026-05-14, 3 reps each)

| skew  | eviction onset (epoch) | regime                          |
|-------|------------------------:|---------------------------------|
| 0.01  | 18                      | wide working set, evicts early  |
| 0.2   | 19                      | same                            |
| 0.4   | 23                      | slightly slower onset           |
| 0.6   | 38                      | tighter hot set, later onset    |
| 0.8   | **never**               | hot set fits entire cache       |
| 0.99  | **never**               | hot set fits entire cache       |

- The clean monotonic mapping of skew -> onset epoch (lower skew
  onsets earlier, then never) demonstrates the cache-locality story
  directly. 1 KB amplifies the effect vs 120 B because each admit
  ships 8x more bytes, so the per-epoch admit cost is dominant
  even before the cache fills.
- At theta>=0.8, the lack of any onset marker indicates the hot set
  fits in cache for the entire run. These curves are effectively
  flat after their initial warmup with no eviction-induced plateau
  transition.

### Caveats
- Single workload (ycsbf), single record size (1 KB). The 120 B
  version of this plot showed faster warmup because 120 B admits
  ship less data per epoch, but had a similar shape; we switched
  to 1 KB to make the eviction onset more visible since admit
  volume crosses cache capacity sooner per epoch.
- Cache ratio (~52.7 %) is the autosizer's natural pick; if you cap
  smaller, all onsets shift earlier; if larger, low-skew may still
  evict but high-skew clearly does not.

### Related
- Plot 03 (uses the same 1KB logs; this plot's per-epoch view
  complements plot 03's windowed-throughput summary)
- Plot 11 (TPCC analog: decline trajectory on tpccfull instead of
  rising on ycsbf)

---

## 13 - Combined cross-workload skew sweep (subsumes plots 04 + 05)

`figures/workload_sweep_combined.{pdf,png,csv}` from `plots/13_workload_sweep_combined.py`

### Headline claim
"In a single 1x4 figure, hybrid_staging is shown against both the
in-HBM ceiling (gpu_only at 5 M records) and the exceeds-HBM floor
(cpu_only at 20 M records) across all four YCSB workloads. The figure
makes the same point as plots 04 and 05 combined, in half the page
space."

### Experimental design
- One row, 4 panels (one per YCSB workload). Same skew sweep on the
  X-axis as plots 04 and 05.
- 4 lines per panel, no new benchmark runs (reads from plot 04's
  `three_way/` logs and plot 05's `beyond_hbm/` logs):

  | Line | Color | Style | Meaning | Source |
  |---|---|---|---|---|
  | `hybrid_staging, 5 M`  | red  | solid  | EGAD, fits in HBM        | plot 04 logs |
  | `gpu_only, 5 M`         | red  | dashed | in-HBM ceiling            | plot 04 logs |
  | `hybrid_staging, 20 M` | blue | solid  | EGAD, exceeds HBM         | plot 05 logs |
  | `cpu_only, 20 M`        | blue | dashed | exceeds-HBM floor         | plot 05 logs |

- Y-axis is shared across panels (sharey=True) so cross-workload
  comparisons read directly.

### How to read
- Per panel: red-pair shows the in-HBM regime (top performance
  region); blue-pair shows the exceeds-HBM regime (lower band).
- Within each color, the solid line is hybrid_staging; the dashed
  line is the corresponding baseline (gpu_only ceiling above, or
  cpu_only floor below the solid hybrid line).
- The gap between red solid and red dashed = "staging machinery cost
  when the DB fits in HBM" (plot 04's story).
- The gap between blue solid and blue dashed = "staging machinery
  advantage when the DB exceeds HBM" (plot 05's story).

### Cite
Numbers come straight from the underlying plot 04 and plot 05 data;
see those entries for the per-cell cite tables. Headline relationships
the combined view makes obvious:
- Red solid (5 M hybrid) sits well above blue solid (20 M hybrid)
  at low skew (the cache-pressure cost is visible per workload).
- The red ceiling (gpu_only at 5 M) and the blue floor (cpu_only at
  20 M) bracket the operating range; hybrid_staging stays in between.
- At theta=0.99 the in-HBM ceiling collapses for ycsba/f
  (GPU-side hot-record contention) and the blue regime catches up.

### Caveats
- Subsumes plots 04 and 05; do not include all three in the same
  paper section. Pick this combined view OR the two separate plots,
  not both.
- The two regimes use DIFFERENT DB sizes (5 M vs 20 M records), so
  reading absolute numbers across the two color groups conflates a
  record-count effect with a cache-pressure effect. The figure's
  legend designates `5 M` vs `20 M` explicitly to keep the
  comparison honest.
- gpu_only at 20 M and cpu_only at 5 M are intentionally absent
  (plot 04 omitted cpu_only because gpu_only is the relevant
  ceiling at small DB; plot 05 omitted gpu_only because the
  hybrid-vs-cpu story is the contribution at large DB).

### Related
- Plot 04 (the in-HBM half, standalone)
- Plot 05 (the exceeds-HBM half, standalone)

---

## 14 - Combined async vs sync per-phase breakdown (subsumes plots 08 + 10)

`figures/breakdown_combined.{pdf,png,csv}` from `plots/14_breakdown_combined.py`

### Headline claim
"Async writeback's gain over sync is set by how close the sync-mode
writeback time is to the rest of the epoch's wallclock, plus the
async path's faster ordered scatter. TPC-C tpccdeck is the optimal
case (writeback exceeds the rest of the pipeline, and the faster
scatter carries the measured 2.28x past the 2x overlap-only bound).
YCSB-F has meaningful writeback but it's smaller than the rest,
giving 1.21x. YCSB-B's writeback is already small (~0.65 ms), so
async and sync are indistinguishable on it (0.98x)."

### Experimental design
- 3 representative workloads x 2 modes = 6 horizontal stacked bars
  in one figure. The workloads are chosen as the representative
  point of each regime:
  - **TPC-C tpccdeck**  -> writeback-heavy, optimal async case
  - **YCSB-F**          -> moderate writeback (represents the
    write-heavy YCSB-A/F pair; A is omitted as a near-duplicate)
  - **YCSB-B**          -> light writeback (represents the
    read-heavy YCSB-B/C pair; C is omitted as a near-duplicate)
- Bar order, top to bottom (biggest async win first): TPC-C tpccdeck,
  YCSB-F, YCSB-B.
- YCSB cells reuse plot 08's logs (`writeback_breakdown/`); TPC-C
  cells reuse plot 10's logs (`tpcc_writeback_breakdown/`). No new
  benchmark runs.
- YCSB's `transfer (compute)` + `transfer (data)` phases are
  aggregated into a single `stage` block to match TPC-C's
  `prepareEpoch` granularity. Both share the same 7-phase axis:
  `index_transfer | indexing | submission | stage | initialization | execution | writeback`.
- Steady-state windows: YCSB e250-e280 (the 300-epoch headline protocol), TPCC e30-e50 (matches plots
  08 and 10 respectively).

### How to read
Per workload, the async bar (top) is shorter than the sync bar (bottom);
the delta is the async win. Within each bar, the red writeback block
shrinks dramatically in async mode (that IS the hiding mechanism). The
sync writeback block is fat on bars where async gains a lot, and thin
on bars where async gains nothing.

Two secondary observations the figure makes visible:
- **stage block grows slightly in async** vs sync - the host-side
  contention tax (Appendix C). On tpccdeck the stage block grows
  from 2.95 ms (sync) to 3.17 ms (async), +8%.
- **index_transfer grows much more in async**: 900 us async vs
  266 us sync on YCSB-F (3.4x), and 1064 us vs 505 us on tpccdeck
  (2.1x). This is the CUDA-runtime contention effect documented in
  Appendix C; without an active writeback worker the inflation does
  not happen, which confirms the worker thread is the cause.

### Cite (TPC-C cells 2026-07-06 epic@996e4a5, e30-50 rep medians; YCSB cells re-measured 2026-08-12, engine 54ca62c, 300-epoch protocol, e250-280 window, rep means; 3 reps each)

| Workload | sync ms | async ms | speedup | wb sync ms | wb async ms |
|---|---|---|---|---|---|
| TPC-C tpccdeck W=128 | 25.65 | **11.24** | **2.28x** | 17.06 (67%) | 1.80 |
| YCSB-F theta=0.5     | 12.49 | 10.29     | 1.21x    | 4.26        | 0.79        |
| YCSB-B theta=0.5     |  8.49 |  8.88     | 0.96x    | 0.64        | 0.41        |

- TPC-C tpccdeck is the **optimal async case** on our workload
  set. The writeback (17.06 ms) exceeds all other phases combined
  (25.65 - 17.06 = 8.59 ms), so overlap hides the non-writeback
  work inside the writeback window, and the async path's faster
  ordered scatter carries the measured 2.28x past the 2x
  overlap-only ceiling.
- YCSB-F has **moderate gain**: writeback ~4.3 ms is smaller
  than the rest of the epoch (~8.2 ms), so most of the writeback
  hides but the savings is bounded by the writeback size, not the
  pipeline.
- YCSB-B has **no meaningful gain**: writeback is already shorter
  than the contention tax, so async is at best a wash.
- (Omitted: YCSB-A behaves like F; YCSB-C behaves like B.
  Including all four would double the plot height without
  changing the story.)

### Caveats
- This figure subsumes plots 08 and 10; do not include all three
  in the same paper section.
- YCSB's `stage` block here aggregates the `transfer (compute)`
  and `transfer (data)` sub-bands separately shown in plot 08.
  If you need the SG-vs-compute split visible, use plot 08
  standalone.
- All cells at theta=0.5 (single skew). At high skew the writeback
  set is smaller so the async win shrinks; this figure shows the
  representative mid-skew case.

### Related
- Plot 08 (YCSB-only breakdown, with `transfer (compute)` /
  `transfer (data)` split visible)
- Plot 10 (TPC-C-only breakdown standalone)
- Appendix C (async writeback host-side contention; now applies to
  both YCSB and TPC-C)

---

## 15 - Cost of GPU-crash recovery (TPCC)

`figures/recovery_cost.{pdf,png,csv}` from `plots/15_recovery_cost.py`

### Headline claim
"Turning on GPU-crash recovery costs hybrid_staging 13-18 % throughput
across W >= 8 (10.5 % at W=4), and recovery-on still beats the
EPIC-CPU floor at every W >= 8 (1.07x to 1.24x). The overhead is
constant in uptime (always exactly two epochs of replay), unlike WAL
whose cost grows with the checkpoint interval."

### Experimental design
- Single mix (`tpccdeck`), e=50, steady-state window e30-50, 3 reps.
- W in {4, 8, 16, 32, 64, 128}.
- Three modes, all on the SAME build (epic@996e4a5, the durable-recovery
  path) so the off->on gap is pure recovery overhead:
  - **off** = non-recovery hybrid_staging, canonical fig-09 config
    (flat indexes, prewarm, parallel sections, autosizer, NUMA node-0,
    OMP 8,3).
  - **on** = provisioned recovery = off + `EPIC_DURABLE_STORE=<dir>`
    (durable Primary Store in /dev/shm, provisioned at load; the
    binary provisions unconditionally since lazy first-touch commit
    measured slower and loses to cpu_only, so no knob exists).
  - **cpu** = stock EPIC cpu_only floor (no env, no NUMA pin, `-c 24`),
    the dashed reference line.
- Recipe in `tests/15_recovery_cost.sh`; durable store cleaned from
  /dev/shm before+after every recovery run (~11 GB at W=8, ~34 GB at
  W=128, both < 63 GB; the store grows with EPOCHS not W, so e=50 fits).

### How to read
- Two bars per W: blue = no recovery, red = with recovery. The red
  drop below blue is the recovery cost, annotated as a percentage.
- Gray dashed EPIC-CPU line is the floor. Red bars sit above it for
  W >= 8; at W=4 both bars sit at the floor (the known low-W region
  where staging overhead ties cpu_only, same as plot 09's W=4 caveat).

### Cite (measured 2026-07-06, epic@996e4a5, 3 reps, e30-50 median)

| W | off (no rec) | on (recovery) | cpu_only | overhead | on/cpu |
|---|----:|----:|----:|----:|----:|
| 4   | 5.52 | 4.94 | 5.79 | 10.5 % | 0.85x |
| 8   | 7.27 | 6.35 | 5.94 | 12.6 % | 1.07x |
| 16  | 8.78 | 7.30 | 5.90 | 16.8 % | 1.24x |
| 32  | 9.24 | 7.54 | 6.06 | 18.4 % | 1.24x |
| 64  | 8.84 | 7.31 | 6.18 | 17.3 % | 1.18x |
| 128 | 8.53 | 6.99 | 6.08 | 18.0 % | 1.15x |

- A re-measure on the final code (durable-mode submit ordering plus the
  delete path active) widens the W>=8 band to 11.8-19.0 %; the paper
  states 12-19 %. The committed figure and this table are the original
  dataset.
- Recovery overhead is 13-18 % across W>=8 (12.6 % at W=8 rising to
  ~17-18 % from W=32 up). The durable stores are tmpfs-backed and do
  not use huge pages, so the off arm's gains widen the percentage
  relative to earlier campaigns; the absolute on-arm throughput
  barely moved.
- The cost is almost entirely durable shadow-index log-append in the
  indexing phase, NOT writeback (the per-epoch shadow scatter was
  removed). It is async-able as future work.
- Recovery-on beats cpu_only at every W>=8; the contribution survives
  with recovery on.

### Caveats
- W=4 is the trivial low end where even non-recovery hybrid ties/loses
  cpu_only (plot 09's W=4 caveat); recovery makes a losing cell lose a
  bit more. The story is the W>=8 band.
- TPC-C only in this figure. YCSB recovery overhead is characterized
  separately at <0.5 % (the YCSB workloads have no steady-state
  inserts, so the shadow-index append cost vanishes).
- off numbers here are a same-campaign re-measure (7.27/8.78/9.24/
  8.84/8.53 for W8-128), consistent with plot 09's hybrid cells
  (7.39/8.72/9.21/8.87/8.56) within ~2 %.

### Related
- Plot 09 (the non-recovery TPC-C warehouse sweep this overhead is
  measured against)

---

## 16 - Epoch-size (batch-size) sensitivity, three systems

`figures/epoch_size_sweep.{pdf,png,csv}` from `plots/16_epoch_size_sweep.py`

### Headline claim
"S=100K, the operating point of every other figure, is the measured
optimum of both Epic baselines and understates EGAD. EPIC-CPU flattens
from 100K up and EPIC-GPU peaks at exactly 100K and declines, while
EGAD keeps gaining through 400K: +28 % from 100K to 400K on YCSB-F
(+9 % then +18 % per doubling) and +9 % on TPC-C. Under Epic's latency
convention (1.5x the epoch wallclock), EGAD is the only system of the
three that keeps converting latency budget into throughput through the
swept range; picking 100K therefore makes every headline comparison
conservative for EGAD."

### Experimental design
- Epoch size S in {5K, 10K, 25K, 50K, 100K, 200K, 400K} for YCSB-F
  (theta=0.5, 20 M x 120 B, `-r true -f false`) and {5K, 25K, 100K,
  400K} for TPC-C deck at W=64 (chosen below the W range's ceiling so
  the S=400K device footprint fits; projected TPC-C table arrays scale
  with total transactions x S).
- Matched total transactions per run (epochs = total/S), so the
  measured window sits at identical table-growth state at every S:
  EGAD YCSB 35.2 M (the long cache warmup is real), stock YCSB 8 M,
  TPC-C 9.6 M. Windows are the LAST 5.28 M / 5 M / 4.8 M transactions.
- Cache capacity pinned independent of S: YCSB hybrid at the canonical
  33.6 % (6.72 M; the autosizer's input subtracts the epoch-
  proportional version arrays, so an unpinned pick would covary with
  S), TPC-C EGAD at the caps a single unpinned S=100K probe chooses,
  pinned via `--cache_capacity_*` on every cell.
- Arms: EGAD = the canonical test-05 / test-09 hybrid invocations.
  YCSB baselines = TRUE UPSTREAM Epic (`setup_epic_stock.sh
  --small-records`; upstream selects modes with `-x` alone). TPC-C
  baselines = this repo's OFF-layout build (upstream has no tpccdeck
  mix; stock-vs-fork TPC-C CPU parity is 0.99-1.00x, see plot 09's
  provenance).
- Latency is reported as 1.5x the mean epoch wallclock, Epic's
  convention; valid for all three systems here because batches are
  formed entirely off the epoch critical path in both harnesses.
- 3 reps per cell, one discarded warm-up run per binary first.
  Recipe in `tests/16_epoch_size_sweep.sh`.

### How to read
- Panels (a) YCSB-F and (b) TPC-C: throughput vs S, log-x, mean +/-
  std over reps. The gray vertical line is S=100K, every other
  figure's operating point. The claim is the SHAPE contrast: both
  baselines peak or flatten at the line, EGAD alone keeps rising
  through the right edge.
- Panel (c): the same YCSB-F data as throughput vs implied latency
  (the Epic-Fig-10 view). The curves are parametric in S, traced from
  5K at each curve's left end to 400K at its right, so the same epoch
  size lands at a different latency per system and faster systems sit
  further left. Rings mark S=100K on every curve, per-curve S labels
  show the sweep direction, and the ticks are plain milliseconds.
  Read it with a latency budget in mind. At any budget, the best a
  system offers is its highest point at or left of that x. EPIC-GPU
  offers its maximum at ~4 ms and loses throughput beyond; EPIC-CPU
  saturates by ~17 ms; EGAD's curve is strictly rising, so at any
  latency target above ~17 ms it is the only system still trading
  latency for throughput.
- EPIC-GPU's absolute level is not the comparison (at 20 M x 120 B the
  whole database fits in HBM, the in-HBM regime of plots 04/13); its
  peak-at-100K shape is.

### Cite (measured 2026-08-09, engine @ 54ca62c, stock upstream @ 5a7dc90 + build patches, 3 reps, mean)

YCSB-F (MTxn/s; EGAD's implied latency in ms in parens):

| S | EGAD | EPIC-CPU | EPIC-GPU |
|---|----:|----:|----:|
| 5K   | 3.39 (2.2)   | 3.39 | 21.91 |
| 10K  | 4.74 (3.2)   | 5.33 | 30.34 |
| 25K  | 6.20 (6.1)   | 7.57 | 38.15 |
| 50K  | 7.69 (9.8)   | 8.10 | 38.25 |
| 100K | 9.37 (16.0)  | 8.64 | **39.62 (peak)** |
| 200K | 10.23 (29.4) | 8.89 | 37.39 |
| 400K | 12.02 (49.9) | 8.95 | 36.00 |

TPC-C deck (W=64, MTxn/s):

| S | EGAD | EPIC-CPU | EPIC-GPU |
|---|----:|----:|----:|
| 5K   | 1.41 | 1.59 | 3.42 |
| 25K  | 5.03 | 4.27 | 12.02 |
| 100K | 9.12 | 6.09 | **19.02 (peak)** |
| 400K | 9.98 | 6.41 | 17.96 |

- EPIC-GPU peaks at exactly 100K on both workloads and declines past
  it (YCSB-F -5.6 % then -3.7 % per doubling; TPC-C -5.6 % from 100K
  to 400K).
- EPIC-CPU is flat from 100K (YCSB-F +2.8 % then +0.7 % per further
  doubling; TPC-C +5.2 % from 100K to 400K).
- EGAD alone keeps rising: YCSB-F +9.2 % then +17.5 % per doubling
  (+28.3 % total), TPC-C +9.4 %.
- EGAD vs EPIC-CPU at theta=0.5: 1.08x at 100K, 1.15x at 200K, 1.34x
  at 400K; below 100K the CPU baseline leads (the low-skew regime of
  plots 04/05/13).
- Operating at 100K therefore understates EGAD's achievable YCSB-F
  throughput by 22 % while sitting at both baselines' optimum.

### Caveats
- The mid-S EGAD YCSB-F cells carry mode structure across reps (std
  up to 0.7 MTxn/s at 200K); the 400K anchor and every baseline cell
  are tight (std <= 0.15). The shape claim rides the tight anchors and
  the baselines' own shapes.
- theta=0.5 is the near-parity low-skew regime for EGAD vs EPIC-CPU
  (the regime framing of plots 04/05/13); even here EGAD pulls ahead
  of EPIC-CPU at large S.
- TPC-C staging cost has no strict steady state at fixed cache (the
  growing O/OL tables deepen the reach-back), so matched totals is the
  controlled protocol; within the fixed 9.6 M total the S>=100K
  windows are quasi-steady.
- Run length and S are memory-coupled on TPC-C (device arrays sized to
  total x S): a doubled-total probe at S=400K exceeds the 32 GB card
  (measurement retained outside the repository). W=64 leaves headroom
  for the swept range.

### Related
- Plots 04/05/13 (the skew-regime story this sweep's theta=0.5 anchor
  belongs to)
- Plot 09 (TPC-C baseline conventions and the W-axis sweep at fixed
  S=100K)

---

## 17 - Per-epoch admission volume across the run (window justification)

`figures/ycsb_admit_warmup.{pdf,png,csv}` from `plots/17_ycsb_admit_warmup.py`
(no new runs; reads the fig-05/13 beyond\_hbm 120 B hybrid logs)

### Headline claim
"The [250,280] measurement window samples a settled regime. Per-epoch
admission volume falls as the hot set accumulates and flattens within
the first tens of epochs at every skew; the slowest decay, theta=0.99,
settles to near-zero admissions by roughly epoch 100."

### How to read
- One line per skew (viridis, dark = low skew), mean across the three
  headline reps; the shaded band is the reporting window.
- Plateau ordering is the expected one: lower skew touches more
  distinct records per epoch, so its steady admission volume is higher
  (theta=0.01 ~0.56 M/epoch; theta=0.99 ~0, the hot set is fully
  cached).
- This is the direct answer to the window-looked-arbitrary concern; the
  companion prose sentence lives in the evaluation setup.

### Cite
- Steady-state admissions (mean over e250-280): 0.56 M/epoch at
  theta=0.01, 0.52 at 0.2, 0.44 at 0.4, 0.31 at 0.6, 0.11 at 0.8,
  0.00 at 0.99.
- Design-section use: each steady admission into a full cache pairs
  with an eviction, so the cache index absorbs up to ~1.1 M single-store
  mutations per epoch against 1 M record operations (100 K txns x 10
  ops), while the key-to-CRID hash table takes zero steady-state
  mutations on insert-free workloads. This is the measured version of
  the two-index argument.

### Caveats
- YCSB-F only (the workload the setup discussion uses); other workloads
  share the admission mechanics.
- Warmup epochs are visible by design; nothing before the window is
  reported anywhere else in the paper.

---

# Appendices

## A. Why 120 B is the headline record size

Decision from the 2026-05-13 pilot (logs retained outside the repository).

At -rT-fF, hybrid beats cpu_only by 2.4x to 3.8x at 120 B across all
workload/skew cells. At 1 KB, hybrid sometimes LOSES to cpu_only at
low skew (the "1 KB ratios collapse" branch of the survey's decision
rule).

The 1 KB sensitivity callout still lives in plot 03 so the paper isn't
making a hidden record-size assumption.

## B. Why "matched cache/DB ratio" matters and which value we use

YCSB performance is fundamentally a function of (read/write semantics,
cache hit rate, skew, record size). Comparing two configurations at
different cache hit rates conflates record-size effects with cache
pressure effects. To isolate per-byte transfer cost from cache pressure,
all hybrid measurements at 120 B + 20 M use a cache cap to match the
natural ratio at the comparison config.

**Two cap values appear across plots:**
- **Plot 03**: `EPIC_YCSB_CACHE_CAP=10544194` (records, ~52.7 % ratio)
  matched against the 1 KB autosizer's natural 10.54 M record pick at
  20 M records.
- **Plots 05, 06, 07, 08**: `EPIC_YCSB_CACHE_CAP=6720000` (records,
  ~33.6 % ratio) matched against the natural 67 M field-units pick at
  -rT-fT / -rF-fT (where the autosizer counts in field-units, capping
  at 33.6 % of 200 M field-units in HBM).

The two cap values produce different absolute throughputs (52.7 % cache
means less eviction than 33.6 % cache), so cross-plot comparisons of
absolute MTxn/s should account for this. Within-plot comparisons are
apples-to-apples by construction.

## C. Async writeback's index_transfer contention

In async mode the previous epoch's writeback worker is still running when
the next epoch's index_transfer H2D starts. The worker fires ~8 chunked
cudaMemcpyAsync + Event D2H bursts on its own non-blocking stream while a
12-thread OMP scatter team writes packed records into node-0 CPU memory
between bursts.

Single-source ablations did NOT inflate index_transfer (D2H alone: 266 us,
scatter alone: 266 us). The combination inflates it to ~963 us:

- **~220 us**: the chunked D2H loop's CUDA API call rate slows the main
  thread's cudaMemcpyAsync and cudaStreamSynchronize through shared
  CUDA-runtime state beyond the public driver lock.
- **~480 us**: the scatter team's cross-NUMA writes from node-1 cores into
  node-0 records saturate the node-0 memory controller, back-pressuring
  the main thread's CUDA API issue path.

The GPU PCIe link and copy engines themselves are not the bottleneck.
This is host-side runtime + memory-controller contention.

**For the paper**: this is a structural cost of the async overlap design,
not a fixable defect. The net win remains substantial: async hides ~3.5 ms
of writeback work and leaks ~0.7 ms back through contention, for a net
~2.8 ms epoch-time savings on write-heavy workloads.

**TPC-C confirms the same effect** on the `stage` block (per plot 14):
tpccdeck async stage 3.77 ms vs sync 3.25 ms (+16 %). The 8 concurrent
TPC-C stagers issuing per-stager prepareEpoch work overlap with the
writeback worker the same way YCSB's main-thread index_transfer does,
so the host-side contention surfaces as a slightly larger `stage`
block in async mode. Same root cause (worker D2H + cross-NUMA scatter
saturate the node-0 memory controller); same conclusion that the
~0.5 ms inflation is dwarfed by the ~11 ms writeback being hidden.

Plot 14 makes this visible on a per-workload basis: index_transfer
inflation appears on every YCSB cell with non-zero writeback (A, B,
F) but is **absent on YCSB-C** (263 us in both modes), confirming
the writeback worker is the cause.

## D. Why -rT-fT was rejected as the pilot headline

Early in the pilot (2026-05-13) we initially set -rT-fT (full-record
reads against field-split storage) as the headline. This was wrong:

- -rT-fT has a 10x field-op amplification (each full-record read
  decomposes into 10 separate field reads under field-split storage).
- At 120 B this dropped hybrid throughput to ~3.7 MTxn/s vs ~13.8 MTxn/s
  at -rT-fF on the same workload.
- Hybrid loses to cpu_only at -rT-fT for theta <= 0.6 (per plot 06).

The historical figure7 hybrid line was at -rF-fT (single-field reads),
which is why "hybrid wins over cpu" held in that dataset while the
-rT-fT pilot showed losses; the two measure different configs.

Resolution: -rT-fF is the headline (the canonical command in
`RUNNING_TESTS.md` §2); -rT-fT is documented as a sensitivity callout
in plot 06. The lesson: before comparing against historical numbers,
decode the exact flags from the historical logs first.
