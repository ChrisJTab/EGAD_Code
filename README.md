# EGAD: Extending GPU-Accelerated Databases

EGAD is a GPU-accelerated transactional database that keeps its primary
store in CPU memory and stages each batch's working set onto the GPU for
execution. Like prior GPU-accelerated OLTP systems, it plans concurrency
control before executing each batch deterministically and uses
dual-versioning. EGAD adds a staging layer that admits the records a
batch needs and writes back the records the batch modifies. Because a
batch's writes only touch one of the two versions, EGAD halves
per-record PCIe traffic, runs writeback concurrently with the next
batch's execution, and keeps the CPU-side Primary Store at most two
batches behind the GPU. On a GPU crash, recovery replays at most two
batches of transaction inputs, since everything older already
resides in CPU memory. The system is built on Epic (OSDI'24), whose
deterministic batched execution it extends beyond device memory, and the
included harness reproduces the paper's YCSB and TPC-C evaluation on
databases that exceed device memory.

The paper describing EGAD is included as [EGAD_paper.pdf](EGAD_paper.pdf).

## Repository layout

| Path | Contents |
|---|---|
| `EGAD/` | The EGAD system itself (a fork of Epic), vendored as a plain directory |
| `EGAD_plotting/` | Figure-reproduction harness. `tests/*.sh` produce logs, `plots/*.py` render figures. See `EGAD_plotting/RUNNING_TESTS.md` (how to run) and `EGAD_plotting/PLOTS.md` (what each figure claims) |
| `validation/` | Recovery and determinism validation (crash/recover sweeps, negative control) |
| `setup_epic_stock.sh`, `patches/` | Reproduces the unmodified-upstream Epic checkout behind the stock CPU baseline cells (pinned clone plus two build fixes; see `EGAD_plotting/PLOTS.md` provenance) |
| `build_binaries.sh` | Canonical build entry point (see Build below) |
| `install_dependencies.sh` | Ubuntu package + CUDA toolchain setup (needs sudo, reboots the machine at the end) |
| `deallocate_hugepages.sh` | Releases a persistent hugepage reservation before benchmarking (see Preflight below) |

## Hardware and software assumptions

The committed figures were produced on this configuration.

- 2-socket Intel Xeon Silver 4410Y, 24 physical cores (48 logical CPUs), 384 GB total (64 GB DDR5-4000 per CPU node plus a 256 GB CPU-less memory node the runs leave unused)
- NVIDIA RTX 5000 Ada (32 GB, CC 8.9), PCIe Gen4 x16, attached to NUMA node 0; only GPU 0 is used
- Ubuntu 24.04, Linux 6.8, gcc 13, CUDA 12.9, CMake 3.28

In practice you need a CUDA GPU with 32 GB of device memory (the
compiled CUDA architectures are 80/86/89), roughly 128 GB of host RAM,
and a known GPU-to-NUMA-node mapping (the harness pins EGAD runs to the
GPU's node). Python 3 with matplotlib renders the figures. Several
harness steps assume passwordless sudo (page-cache drops between cells).

## Build

Clone with submodules and build through the wrapper script. Do not run
cmake by hand for these binaries; the script sets both CMake flags
explicitly on every invocation so stale cache values can never leak
between configurations.

```
git clone --recurse-submodules https://github.com/ChrisJTab/EGAD_Code.git
cd EGAD_Code
./build_binaries.sh --hybrid
```

The record layout is a compile-time flag, and the cpu_only baseline is
value-incorrect on the hybrid layout, so EGAD and the baselines live in
separate build trees.

| Command | Output | Layout | Used for |
|---|---|---|---|
| `./build_binaries.sh --hybrid` | `EGAD/build/epic_driver` | ON, 1 KB YCSB + TPC-C | EGAD (hybrid_staging), all TPC-C tests, 1 KB YCSB |
| `./build_binaries.sh --hybrid --small-records` | `EGAD/build-small/epic_driver` | ON, 120 B YCSB | YCSB hybrid at 120 B |
| `./build_binaries.sh` | `EGAD/build-off/epic_driver` | OFF (stock Epic) | TPC-C cpu_only + gpu_only baselines |
| `./build_binaries.sh --small-records` | `EGAD/build-small-off/epic_driver` | OFF, 120 B YCSB | 120 B YCSB baselines |

## Box preflight (before any benchmarking)

`EGAD_plotting/run_all.sh` enforces the first two checks and refuses to
run otherwise. Check all of them after any reboot; details and the
reasoning are in `EGAD_plotting/RUNNING_TESTS.md` §0.

1. Hugepages released. `HugePages_Total` must be 0 (`grep HugePages_Total /proc/meminfo`); if not, run `./deallocate_hugepages.sh`.
2. CPU frequency governor set to `performance` on all CPUs. A reboot resets it to `schedutil`, which costs up to ~19% throughput on some configurations.
3. GPU 0 idle (no compute apps, under 100 MiB of device memory in use).
4. Quiet box; the tests drop the page cache before every cell (passwordless sudo).

## Reproducing the figures

```
cd EGAD_plotting
./run_all.sh
```

`run_all.sh` runs the preflight, every `tests/*.sh` in order, then every
renderer in `plots/`. Figures and per-cell median CSVs land in
`EGAD_plotting/figures/`. Every cell is skip-if-log-exists, so an
interrupted campaign resumes where it stopped, and `FORCE_RERUN=1`
redoes cells. A full campaign takes days on the reference box (test 09
alone is 153 cells), while a single figure can be reproduced by running
its `tests/NN_*.sh` followed by its `plots/NN_*.py`.

`EGAD_plotting/PLOTS.md` maps every figure to its claim, experimental
design, citable numbers, and caveats. `EGAD_plotting/RUNNING_TESTS.md`
documents the canonical invocation for every mode (binary, env vars,
NUMA pinning, epoch windows) plus a failure-triage table.

## Recovery validation

Crash/recover correctness runs outside the figure harness, on a build
with the validation hooks compiled in (the shipped binaries exclude
them and reproduce the non-recovery headline numbers).

```
cmake -S EGAD -B EGAD/build-val -DCMAKE_BUILD_TYPE=Release \
      -DHYBRID_RECORD_LAYOUT=ON -DEGAD_VALIDATION=ON
cmake --build EGAD/build-val -j

BIN=EGAD/build-val/epic_driver bash validation/crash_recover_sweep.sh
BIN=EGAD/build-val/epic_driver bash validation/ycsb_crash_recover_sweep.sh
BIN=EGAD/build-val/epic_driver bash validation/negative_control.sh
```

The sweeps inject a real GPU fault mid-run (the process dies), restart,
recover from the durable store, and compare a placement-invariant state
hash against a no-crash baseline. The negative control proves the hash
gate detects corruption rather than passing blindly. See
`validation/README.md`.

## License

To be determined before public release.

## Citation

Citation information will be added when the paper is published.
