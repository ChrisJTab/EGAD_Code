# EGAD validation harness

Reproducibility scripts for the recovery and determinism claims in the paper.
These are **not** part of the default product build. The fault injector,
state-hash, and negative-control hooks they drive are gated behind the
`EGAD_VALIDATION` compile flag (Phase 9), so the shipped `epic_driver` excludes
them and reproduces the non-recovery headline numbers unchanged.

## Build the validation binary first

The scripts will neither inject a GPU fault nor emit `[STATE-HASH]` lines on the
default build. Configure a binary with `-DEGAD_VALIDATION=ON`:

```
cmake -S EGAD -B EGAD/build-val \
      -DCMAKE_BUILD_TYPE=Release -DHYBRID_RECORD_LAYOUT=ON -DEGAD_VALIDATION=ON
cmake --build EGAD/build-val -j
```

The scripts default `BIN` to `EGAD/build/epic_driver`. Either rebuild `EGAD/build`
with `-DEGAD_VALIDATION=ON`, or point the script at the validation binary
(`BIN=EGAD/build-val/epic_driver bash validation/<script>.sh`).

GPU + a writable durable store directory (we use `/dev/shm`) are required; the
scripts clean it before and after. Check GPU tenancy first
(`nvidia-smi --query-compute-apps`).

## Scripts and the claims they reproduce

| script | paper claim | what it checks |
|---|---|---|
| `crash_recover_sweep.sh` | TPC-C GPU-crash recovery (design §4.9, eval recovery subsection) | For each (crash epoch, crash phase), a durable worker injects a **real** GPU fault (illegal access → CUDA err 700 → process dies non-zero); a fresh process re-maps the durable Primary Store, rolls back to end-of-(E-2), replays E-1 and E, and finishes the run. PASS = recovered placement-invariant `[STATE-HASH-VALMS]` equals the no-crash baseline (logically identical state). |
| `ycsb_crash_recover_sweep.sh` | YCSB GPU-crash recovery | Same real-fault crash/recover sweep for YCSB; gate is `[STATE-HASH-VALONLY]`. |
| `negative_control.sh` | The recovery gate is sensitive (the PASSes are meaningful) | Without a crash, perturbs the final Primary Store via the gated `EPIC_RECOVERY_CORRUPT_ONE` / `EPIC_RECOVERY_SWAP_TWO` hooks. PASS = a 1-byte corrupt **diverges** both the value and positional hashes, and a two-record full-value swap **diverges** the positional gate. Proves the recovery hash gate catches corruption rather than passing blindly. |

## Notes

- The gate compares **logical** content (`[STATE-HASH-VALMS]` / `[STATE-HASH-VALONLY]`),
  which is invariant to internal CRID placement and dual-version slot choice, since
  those are legitimately non-deterministic for inserts under eviction. The positional
  `[STATE-HASH]` is the stricter byte-placement hash used by the negative control.
- Determinism requires the deterministic seed (`EPIC_TPCC_SEED` / `EPIC_YCSB_SEED`),
  also gated behind `EGAD_VALIDATION`. The production recovery path itself does **not**
  depend on the seed: the recover process regenerates its transaction inputs (there is
  no durable input log), rolls the durable store back to a consistent epoch boundary,
  and resumes. Without the seed the resumed transactions are fresh rather than a replay
  of the crashed run's, so only the byte-for-byte *paired-run comparison* needs the seed.
