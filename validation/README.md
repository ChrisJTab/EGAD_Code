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
| `ycsb_crash_recover_sweep.sh` with `WL=ycsbw` / `WL=ycsbx` | Deletes under recovery (design, deletes subsection) | The delete workloads (sliding-window churn; the adversarial ycsbx mix that reads, writes, deletes and re-inserts the same keys inside one epoch and across epochs) run with the cache capped so eviction and the reclaim-first pass execute, with the serial-order oracle on (`EPIC_TIMELINE_ORACLE=1`, see below), and with the drop-one-delete negative control. Extra gates: `[STATE-HASH-LIVE]`, `[LIVE-CHECK]`, `[TIMELINE-ORACLE]`, `[MAP-CHECK]`. |
| `delete_semantics_check.sh` | Deletes and inserts are resolved in serial order (design, deletes subsection) | Runs ycsbx with the host oracle: every operation's resolved record must equal a serial-order replay of the epoch, and the GPU index must equal the model's live set at the end (`[MAP-CHECK]`). Then proves the gate is sensitive: `EPIC_TIMELINE_ORDER_BLIND=1` resolves the way epoch-boundary deletes would, and the oracle must fail. Then runs ycsbw under the oracle. |
| `negative_control.sh` | The recovery gate is sensitive (the PASSes are meaningful) | Without a crash, perturbs the final Primary Store via the gated `EPIC_RECOVERY_CORRUPT_ONE` / `EPIC_RECOVERY_SWAP_TWO` hooks. PASS = a 1-byte corrupt **diverges** both the value and positional hashes, and a two-record full-value swap **diverges** the positional gate. Proves the recovery hash gate catches corruption rather than passing blindly. |

## Validation-build hooks used by these scripts

- `EPIC_TIMELINE_ORACLE=1`: host replay of every epoch in serial order, compared per operation with the GPU's resolved
  records; `[TIMELINE-ORACLE]` per epoch and a final PASS/FAILED line; the end-of-run `[MAP-CHECK]` compares the GPU index
  with the model's live set (YCSB) or with the live set implied by the durable logs (TPC-C NewOrder).
- `EPIC_TIMELINE_ORDER_BLIND=1`: resolves operations without regard to their serial position (a control; the oracle must fail).
- `EPIC_TIMELINE_FORCE_GENERAL=1`: every table resolves through the timeline path, so workloads without deletes or duplicate
  inserts exercise it and must reproduce the fast path's hashes exactly.
- `EPIC_RECOVERY_DROP_ONE_DELETE=1`: suppresses one delete-log entry on recover (the `[LIVE-CHECK]` gate must trip).

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
