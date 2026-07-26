#!/usr/bin/env bash
# TPC-C real-fault crash-and-recover sweep. For each (crash
# epoch CE, crash phase CP): run a DURABLE worker that injects a REAL GPU fault
# (illegal access -> err 700 -> process dies non-zero), then a FRESH recover
# process that re-maps the durable Primary Store, rolls back to end-of-(E-2),
# replays E-1 and E, and finishes all epochs. The recover run's placement-
# invariant [STATE-HASH-VALMS] must equal the no-crash baseline. Mirrors the
# YCSB within-epoch sweep. Correctness is co-tenant-safe, but check tenancy.
#
# Phases: 0=top-of-epoch (before runEpoch), 1=after staging, 2=after execution,
#         3=after E's writeback queued (marker E+1, bytes landing),
#         4=E-1 drained, before the marker bump (marker E, none of E durable),
#         5=after the bump, before E's writeback starts (marker E+1, none durable).
#
# Usage:
#   GPU=2 W=8 BASELINE=0x6575e012780a61cc CES="10" CPS="0" bash crash_recover_sweep.sh
#   GPU=2 W=8 CES="2 10 18 19" CPS="0 1 2 3 4 5" bash crash_recover_sweep.sh   # full sweep
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Default build excludes the validation hooks (Phase 9); point BIN at a
# -DEGAD_VALIDATION build (e.g. EGAD/build-val/epic_driver) to run this harness.
BIN=${BIN:-EGAD/build/epic_driver}
GPU=${GPU:-2}
W=${W:-8}
SEED=${SEED:-42}
Z=${Z:-true}
BASELINE=${BASELINE:-}
BASEPOS=${BASEPOS:-}
CES=${CES:-"10"}
CPS=${CPS:-"0"}
EPOCHS=${EPOCHS:-20}
DIR=${DIR:-/dev/shm/egad_tpcc_cr}

NUMA=(numactl --physcpubind=12-23,36-47 --membind=1)
BASEENV=(EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1
         EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1
         EPIC_WORKLOAD_AWARE_AUTOSIZER=1
         OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false
         CUDA_VISIBLE_DEVICES=$GPU EPIC_TPCC_SEED=$SEED)
ARGS=(-b tpccdeck -d epic -w "$W" -a 0.0 -r false -c 24 -s 100000 -f false -m false
      -n 20000000 -x gpu -e "$EPOCHS" -y hybrid_staging -z "$Z" --verify_tpcc)
valms () { grep -aoE 'STATE-HASH-VALMS.*0x[0-9a-f]+' "$1" | grep -oE '0x[0-9a-f]+' | tail -1; }
# positional [STATE-HASH] (byte-exact placement). Recovery reproduces it
# byte-identically, so gating on it too catches value-swaps the value-multiset misses.
pos () { grep -aoE 'STATE-HASH\] .*0x[0-9a-f]+' "$1" | grep -oE '0x[0-9a-f]+' | tail -1; }

# Auto-compute the no-crash baseline for this (W,Z,SEED) if not supplied, so each
# config self-validates against its own normal-mode run (durability off).
if [ -z "$BASELINE" ]; then
    blog=/tmp/cr_baseline_w${W}_z${Z}_s${SEED}.log
    env "${BASEENV[@]}" "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "$blog" 2>&1
    BASELINE=$(valms "$blog")
    BASEPOS=$(pos "$blog")
    echo "[baseline] no-crash W=$W z=$Z seed=$SEED -> VALMS=${BASELINE:-<none>} POS=${BASEPOS:-<none>}"
fi

pass=0; fail=0; failed=""
echo "Crash-recover sweep: GPU=$GPU W=$W z=$Z seed=$SEED epochs=$EPOCHS baseline=$BASELINE CES={$CES} CPS={$CPS}"
echo "----------------------------------------------------------------------"
for ce in $CES; do for cp in $CPS; do
    rm -rf "$DIR"; mkdir -p "$DIR"
    wlog=/tmp/cr_w_e${ce}_p${cp}.log
    rlog=/tmp/cr_r_e${ce}_p${cp}.log
    # worker: durable store + inject real fault at (ce, cp). Expected to die non-zero.
    env "${BASEENV[@]}" EPIC_DURABLE_STORE="$DIR" EPIC_CRASH_AT_EPOCH="$ce" EPIC_CRASH_AT_PHASE="$cp" \
        "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "$wlog" 2>&1
    wrc=$?
    # recover: fresh process re-maps the durable store and replays E-1, E.
    env "${BASEENV[@]}" EPIC_RECOVER_FROM="$DIR" \
        "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "$rlog" 2>&1
    rrc=$?
    rvm=$(valms "$rlog"); rpos=$(pos "$rlog")
    vfail=$(grep -aE '\[VERIFY\].*FAILED' "$rlog")
    demoted=$(grep -aoE 'demoted [0-9]+ slots' "$rlog" | grep -oE '[0-9]+' | head -1)
    ok=1; why=""
    { [ "$rvm" = "$BASELINE" ] && [ -n "$rvm" ]; } || { ok=0; why+=" VALMS(${rvm:-none}!=$BASELINE)"; }
    if [ -n "$BASEPOS" ]; then { [ "$rpos" = "$BASEPOS" ] && [ -n "$rpos" ]; } || { ok=0; why+=" POS(${rpos:-none}!=$BASEPOS)"; }; fi
    [ -n "$vfail" ] && { ok=0; why+=" [VERIFY]FAILED"; }
    if [ "$ok" = 1 ]; then
        echo "  PASS  crash@e${ce} p${cp}  worker_rc=$wrc recover_rc=$rrc demoted=${demoted:-?}  VALMS=$rvm POS=$rpos"
        pass=$((pass+1))
    else
        echo "  FAIL  crash@e${ce} p${cp}  worker_rc=$wrc recover_rc=$rrc diverged:$why  (logs: $wlog $rlog)"
        fail=$((fail+1)); failed+=" e${ce}p${cp}"
    fi
    rm -rf "$DIR"
done; done
echo "----------------------------------------------------------------------"
echo "RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "ALL CRASH-RECOVER CELLS MATCH BASELINE (VALMS)" || echo "FAILED:$failed"
