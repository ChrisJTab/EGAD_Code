# EGAD engine

This directory holds the EGAD system itself, a fork of Epic (OSDI '24)
that extends its deterministic batched execution to databases larger
than GPU memory. The fork adds a two-store design (a CPU-resident
Primary Store and a GPU Execution Cache), per-table stagers that admit
each batch's working set and write modifications back asynchronously,
GRID-addressed hybrid executors for YCSB and TPC-C, and GPU-crash
recovery (durable Primary Store, dual-version rollback, shadow indexes,
two-batch replay). Epic's original execution modes remain available:
`gpu_only` runs everything in device memory and `cpu_only` executes on
the CPU.

Build through `../build_binaries.sh`, which produces the performance
binaries (hybrid or stock record layout, two YCSB record sizes) in
per-configuration `build*/` directories. The record layout is a
compile-time flag and the `cpu_only` baseline is value-correct only on
the stock layout, so the configurations must not share a build tree; do
not run cmake by hand for these binaries. A fifth configuration adds
`-DEGAD_VALIDATION=ON` and compiles the crash-injection and state-hash
harness that `../validation/` drives; the flag is off in every
performance binary.

Run conventions, per-mode invocations, and the figure harness are
documented in `../EGAD_plotting/RUNNING_TESTS.md`; recovery validation
in `../validation/README.md`.

Upstream Epic is BSD 3-Clause licensed and the fork retains its license
(see `LICENSE`).
