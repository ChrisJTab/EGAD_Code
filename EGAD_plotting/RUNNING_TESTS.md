# Running the EGAD plot harness

How to run every test in `tests/` correctly: which binary, which env vars, NUMA
or not, how many cores, which epochs, and where the steady-state window sits.
This file is about *execution*; `PLOTS.md` documents what each figure claims and
cites. It supersedes the root `RUNNING_HYBRID_STAGER.md` for harness purposes.

Every command below is exactly what the test scripts run. If you change a
convention, change the test script and this file together.

---

## 0. Box preflight (before ANY benchmarking session)

`run_all.sh` enforces the first two automatically (Phase 0) and refuses to run
otherwise. Check all of them after any reboot:

| Check | Must be | Command | Why |
|---|---|---|---|
| Hugepages | `HugePages_Total: 0` | `grep HugePages_Total /proc/meminfo`; if not 0, run `../deallocate_hugepages.sh` | A reboot re-applies `vm.nr_hugepages=50000` from sysctl.conf (100 GB locked, node 0 loses 34 GB of its 64 GB). `membind=0` runs then decay with epoch depth. This silently corrupted three weeks of runs once (2026-06-18 to 07-05). |
| Governor | `performance` on all 48 CPUs | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`; fix with `for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance \| sudo tee $c; done` | Reboot resets to schedutil. |
| GPU 0 idle | no compute apps, < 100 MiB used | `nvidia-smi --id=0 --query-compute-apps=pid --format=csv` | Leftover memory inflates footprints and starves the hybrid autosizer. After a *crashed* cell, a core-dumping corpse can hold device memory for tens of seconds; wait for `memory.used` < 100 MiB before the next cell. |
| Quiet box | loadavg near 0, node 0 free ~62 GB | `cat /proc/loadavg; numactl -H` | cpu_only is memory-bandwidth bound and noisy. |
| THP available | `THP_enabled: 1` (or the line absent) | `grep THP_enabled /proc/self/status`; if 0, prefix the launch with `tools/thp_on` | Some launch environments (containers, sandboxes, tuned latency profiles) set `prctl(PR_SET_THP_DISABLE)`; children inherit it, `madvise(MADV_HUGEPAGE)` silently no-ops, and the TPC-C growing-store huge pages never materialize (hybrid deck/NP W=128 lose 6-12% vs a normal shell). `run_all.sh` refuses to start when the flag is set. |
| THP defrag | `defer` | `cat /sys/kernel/mm/transparent_hugepage/defrag`; fix with `echo defer \| sudo tee /sys/kernel/mm/transparent_hugepage/defrag` | A reboot restores the `madvise` default, under which growing-store huge-page faults do synchronous direct compaction once node 0's free 2 MB-contiguous blocks fragment away (~e285 on the e=400 cliff cells): `compact_stall` storms inside the flush worker and deck W=128 steps 8.3 → 2.9 MTxn/s. `defer` falls back to 4 KB pages instead (deep runs taper ~8%, no cliff); e ≤ 50 cells are insensitive to the setting (verified identical windows). `run_all.sh` refuses to start otherwise. |
| Page cache | dropped per cell | tests do `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`; needs passwordless sudo | First-touch consistency for the pageable Primary Store. |

Hardware map: GPU 0 sits on NUMA node 0 (cores 0-11, 24-35; 64 GB). Node 1 has
cores 12-23, 36-47. Node 2 is a 262 GB CPU-less memory node; nothing should
land there during figure runs.

---

## 1. Binaries (the two-binary discipline)

`HYBRID_RECORD_LAYOUT` is a compile-time flag, so each (layout, YCSB record
size) pair is its own build directory. Always build via `./build_binaries.sh`
from the repo root (it sets both CMake flags explicitly every time, and each
combination has its own directory, so builds never clobber each other). Never
run cmake by hand.

| Command | Directory | Layout | Used for |
|---|---|---|---|
| `./build_binaries.sh --hybrid` | `EGAD/build/` | ON, 1 KB YCSB + all TPCC | hybrid_staging (all TPCC tests, 1 KB YCSB) |
| `./build_binaries.sh --hybrid --small-records` | `EGAD/build-small/` | ON, 120 B YCSB | YCSB hybrid_staging at 120 B |
| `./build_binaries.sh` | `EGAD/build-off/` | OFF (stock Epic) | TPCC cpu_only + gpu_only baselines |
| `./build_binaries.sh --small-records` | `EGAD/build-small-off/` | OFF, 120 B YCSB | 120 B YCSB baselines (pending migration, see below) |

Why the split matters:

- **cpu_only is value-incorrect on the ON layout.** The CPU executor loads both
  version tags with one 64-bit atomic that assumes they are adjacent; ON moves
  them ~128 B apart, so it silently reads the wrong value slot. cpu_only MUST
  run on an OFF build.
- **gpu_only is value-correct on ON but the padding roughly doubles small TPC-C
  records**, inflating its device footprint and producing fake OOM ceilings
  (the old "cannot run at W=128" was this artifact; stock OFF runs to W=216).
- hybrid_staging requires ON.

Every test now selects the OFF builds for its cpu_only / gpu_only cells
(migrated 2026-07-06): tests 03/05/06 pick `build-small-off` (or `build-off`
for the 1 KB side) for cpu_only, test 04 runs gpu_only on `build-small-off`,
test 07 runs both reference lines on `build-small-off`, and test 11's
cpu_only floor cell uses `build-off`. Tests 09 and 15 were already correct.

After a fresh build, run 3-5 throwaway warmup reps before any figure cell; the
first few runs on a new binary are ~5% slower.

---

## 2. Canonical invocations, per mode

Throughput is always computed from the `Running epoch N` log timestamps as
`(b - a) * 100000 / (t_b - t_a) / 1e6` MTxn/s over the steady-state window
`[a, b]`, then the median across 3 reps. Windows per mode:

| Mode | Epochs | Window |
|---|---|---|
| TPCC, all three modes | 50 (test 11: 400) | e30-50 |
| YCSB hybrid_staging | 300 | e250-280 (long warmup is real; earlier windows over-report) |
| YCSB gpu_only | 5 | e2-5 |
| YCSB cpu_only | 15 | e3-10 (plot 05 uses e2-5; cpu_only is steady from ~e2 so both work) |

### TPCC hybrid_staging (ON binary, NUMA node 0)

```
EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1 \
EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1 \
EPIC_PREPARE_EPOCH_PARALLEL=1 EPIC_WORKLOAD_AWARE_AUTOSIZER=1 \
OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false \
CUDA_VISIBLE_DEVICES=0 \
numactl --physcpubind=0-11,24-35 --membind=0 \
  EGAD/build/epic_driver -b {tpccdeck|tpcc|tpccfull} -d epic -w $W -a 0.0 \
    -r false -c 24 -s 100000 -f false -m false -n 20000000 \
    -x gpu -e 50 -y hybrid_staging -z true
```

- Pin to node 0 (GPU 0's node): cores AND memory (`--membind=0`).
- `OMP_NUM_THREADS=8,3` is the nested 8x3 team layout for the parallel
  per-stager prepare; keep all four OMP vars together.
- The 8 `EPIC_*` vars are hybrid-mode defaults on phase9+ binaries; the harness
  still sets them explicitly so older commits reproduce identically.
- `-z true` = async writeback (the headline config). `-z false` only in the
  breakdown tests (08/10/14).
- HBM reserve defaults to 4 GB; tests 01 and 11 pass
  `--hybrid_hbm_reserve_gb=6` deliberately (undersized-cache demonstrations).
- Recovery cells add `EPIC_DURABLE_STORE=<dir>` (equivalently `--durable_store`
  on phase9+); clean the durable dir before and after every run.

### TPCC cpu_only (OFF binary, stock, as-shipped)

```
CUDA_VISIBLE_DEVICES=0 \
  EGAD/build-off/epic_driver -b {tpccdeck|tpcc} -d epic -w $W -a 0.0 \
    -r false -c 24 -s 100000 -f false -m false -n 20000000 \
    -x cpu -e 50 -y cpu_only
```

- NO env vars, NO numactl, ever. The paper states EPIC-CPU runs as shipped;
  NUMA placement is part of EGAD's design, not Epic's.
- `-x cpu` is mandatory; `-x gpu -y cpu_only` segfaults (TxnBridge aliases GPU
  memory).
- `-c 24` = one thread per physical core; more threads is slower on this box.
- Noisiest mode (±5-8% run to run). 3-rep medians minimum; for A/B claims,
  interleave the two sides rep by rep instead of running blocks.
- It still uses the GPU (indexing/init + insert pools live there), so it needs
  GPU 0 free, and very long runs can exhaust device memory (pools scale with
  `-e`; W=128 fails by e300).

### TPCC gpu_only (OFF binary, no NUMA)

```
EPIC_FLAT_AUX_INDEX=1 EPIC_FLAT_INDEX_OL=1 EPIC_MIX_REALISTIC_SIZING=1 \
CUDA_VISIBLE_DEVICES=0 \
  EGAD/build-off/epic_driver -b {tpccdeck|tpcc} -d epic -w $W -a 0.0 \
    -r false -c 32 -s 100000 -f false -m false -n 20000000 \
    -x gpu -e 50 -y gpu_only
```

- The 3 env vars give it the same flat index structures and realistic pool
  sizing as hybrid (fairness); no staging vars, no `-z`.
- Expected OOM boundaries at e=50 on the 32 GB card: tpccdeck alive through
  W=216, NP (tpcc) through W=208. Test 09's NP cells at 216/240 abort with
  rc=134 *by design* and render as the line ending at 208.
- Pools are provisioned for the whole run, so the ceiling drops as `-e` grows
  (deck W=128 fits at e=100, fails at e=125).

### YCSB hybrid_staging (build-small for 120 B, build for 1 KB)

```
EPIC_YCSB_CACHE_CAP=<cap> CUDA_VISIBLE_DEVICES=0 \
OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
numactl --physcpubind=0-11,24-35 --membind=0 \
  EGAD/build-small/epic_driver -b ycsb{a|b|c|f} -d epic -w 1 -a $skew \
    -r true -c 32 -s 100000 -f false -m false -n 20000000 \
    -x gpu -e 300 -y hybrid_staging -z true
```

- Cache-cap conventions: `6720000` = 33.6% of 20 M (headline; plots 05/06/08);
  `10544194` = 52.7% (plot 03's matched record-size ratio, 120 B side only);
  sweep values in plot 07; unset = autosizer (natural pick).
- `-r`/`-f` select read width / field splitting; headline is `-r true -f false`
  (rT-fF). Plot 02 uses rF-fT, plot 06 sweeps the combinations.
- 5 M-record cells (plot 04) drop the cap and use `-n 5000000`.

### YCSB cpu_only and gpu_only

```
cpu_only:  CUDA_VISIBLE_DEVICES=0 <binary> ... -c 24 -x cpu -e 15 -y cpu_only -z true
gpu_only:  CUDA_VISIBLE_DEVICES=0 OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
           numactl --physcpubind=0-11,24-35 --membind=0 \
           <binary> ... -c 32 -x gpu -e 5 -y gpu_only -z true
```

- Currently `<binary>` = `build-small` (ON) in tests 03-08; see the pending
  OFF-migration note in §1.
- No cache cap for either (neither has EGAD's cache).

---

## 3. Harness mechanics

- `./run_all.sh` runs the preflight, then every `tests/*.sh` in lexicographic
  order, then every `plots/*` renderer. `EGAD_LOGS_DIR` / `EGAD_FIGURES_DIR`
  are exported by it; the tests also work standalone if you export
  `EGAD_LOGS_DIR` yourself.
- Every cell is skip-if-log-exists; `FORCE_RERUN=1` redoes them. Do NOT
  `FORCE_RERUN` over figure-era logs: the logs are gitignored and exist nowhere
  else. Move the directory aside first
  (`mv logs/<name> logs/<name>.pre_<tag>`) and let the harness regenerate.
- One GPU, strictly serial. Never run two cells at once, and after any rc != 0
  cell give the corpse a few seconds (or poll `nvidia-smi`) before launching
  the next; a core-dumping process holds device memory while it dies, and the
  next hybrid cell's autosizer will read that as a tiny budget and throw.
- Run long campaigns via `nohup`/background and keep the box otherwise idle;
  don't build while cells run.
- Figures pin to an epic commit. Record which commit `build*/` was built from
  when you commit regenerated figures (current dataset: `phase9-cleanup@19a1812`,
  both binaries).

---

## 4. Per-test reference

| Test | What it produces | Cells | Caveats beyond §2 |
|---|---|---|---|
| 01 tpcc_overlap | single deck W=128 e50 hybrid log for the overlap timeline | 1 | uses `--hybrid_hbm_reserve_gb=6` |
| 02 ycsb_overlap | single YCSB-F 120 B rF-fT e300 hybrid log | 1 | rF-fT, not the headline config |
| 03 record_size | 120 B vs 1 KB, hybrid vs cpu_only, 6 skews | 72 | 120 B on `build-small`, 1 KB on `build`; cap `10544194` on 120 B hybrid only (matched 52.7%) |
| 04 three_way | 5 M records (fits in HBM), hybrid vs gpu_only, 4 workloads | 144 | `-n 5000000`, no cap; gpu_only e5 |
| 05 beyond_hbm | 20 M @ 33.6%, hybrid vs cpu_only, 4 workloads | 144 | cap `6720000` on hybrid only |
| 06 config_sensitivity | rT-fF / rT-fT / rF-fT, hybrid vs cpu_only | 108 | cap `6720000` only on rT-fF (others autosize to ~33.6% naturally) |
| 07 cache_sensitivity | cache-ratio sweep 5-100% + gpu/cpu reference lines | 24 | refs measured once (3 reps each), hybrid e300 per cap |
| 08 writeback_breakdown | async vs sync per-phase decomposition, YCSB | 24 | e=40, phase parse over e30-40; `-z false` for sync cells |
| 09 tpcc_warehouse_sweep | headline TPC-C figure; deck W{4..128}, NP W{4..240}, 3 modes | 153 | ON for hybrid, OFF for cpu+gpu; NP gpu cells at 216/240 fail rc=134 by design; W<=2 excluded (pre-existing instability); a W=4 cpu_only cell can rarely segfault (rc=139), delete its log and rerun that cell |
| 10 tpcc_writeback_breakdown | TPCC async vs sync decomposition, deck W=128 | 6 | e50, ON binary |
| 11 tpcc_cliff | deep-run cliff demo, e=400, reserve 6 GB, deck (flat) vs tpccfull (cliffs ~e133) | 7 | cpu floor cell on `build-off` per §1; e400 runs take minutes each |
| 12-14 | plot-only (warmup trajectory, combined workload sweep, combined breakdown) | 0 | reuse logs from 03/04/05/08/10; run those first |
| 15 recovery_cost | recovery on/off/cpu across deck W{4..128} | 54 | durable store in `/dev/shm/egad_rec_cost` (~34 GB at W=128), cleaned before+after each `on` cell; cpu on OFF binary |

Recovery *correctness* (crash/recover byte-identical sweeps) is not in this
harness; it lives in `../validation/` and needs a `-DEGAD_VALIDATION` build.

---

## 5. Failure triage

| Symptom | Meaning |
|---|---|
| rc=134, `cuco`/`thrust` `cudaErrorMemoryAllocation` during setup | Device memory genuinely exhausted. Expected for gpu_only past its W ceiling, or any mode with large `-e` (pools scale with run length). |
| rc=139 (SIGSEGV) at W <= 4, cpu_only | Pre-existing low-warehouse instability, rare and intermittent. Rerun the cell. |
| Hybrid throws `[STAGING-AUTOSIZE-WL] Static tables alone ... exceed HBM budget` at a W that normally works | Something else was holding GPU memory when the autosizer sampled (usually the previous crashed cell still core-dumping). Check tenancy, wait, rerun. |
| Hybrid throughput decays over epochs; node 0 free is low | Hugepage reservation is back (reboot). Preflight (§0). |
| cpu_only SIGSEGV immediately | `-x gpu` was passed; use `-x cpu`. |
| First reps after a rebuild ~5% slow | Warmup effect; discard 3-5 runs. |
| `sudo: a password is required` noise in logs | drop_caches skipped (harness tolerates it); fix passwordless sudo for figure runs. |
