#!/usr/bin/env bash
# YCSB real-fault crash-and-recover sweep. For each (crash epoch
# CE, crash phase CP): run a DURABLE worker that injects a REAL GPU fault (illegal
# access -> err 700 -> process dies non-zero), then a FRESH recover process that
# re-maps the durable Primary Store, rolls back to end-of-(E-2), replays E-1 and E,
# and finishes all epochs. The recover run's placement-invariant [STATE-HASH-VALONLY]
# must equal the no-crash baseline. Mirrors the TPC-C sweep.
# Correctness is co-tenant-safe, but check tenancy.
#
# Phases: 0=top-of-epoch (before runEpoch), 1=after staging/admission, 2=after
#         execution, 3=after E's writeback queued (marker E+1, bytes landing),
#         4=E-1 drained, before the marker bump (marker E, none of E durable),
#         5=after the bump, before E's writeback starts (marker E+1, none durable).
#
# Usage:
#   GPU=2 WL=ycsbf CES="10" CPS="0 1 2 3 4 5" bash ycsb_crash_recover_sweep.sh
#   GPU=2 WL=ycsbi CES="2 5 7" CPS="0 1 2 3 4 5" bash ycsb_crash_recover_sweep.sh
#   GPU=2 WL=ycsbf CES="2 10 19" CPS="0 1 2 3 4 5" bash ycsb_crash_recover_sweep.sh  # full
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=${BIN:-EGAD/build/epic_driver}
GPU=${GPU:-2}
WL=${WL:-ycsbf}              # ycsbf (no inserts) | ycsbi (insert workload)
SEED=${SEED:-42}
Z=${Z:-true}
BASELINE=${BASELINE:-}
BASEPOS=${BASEPOS:-}
CES=${CES:-"10"}
CPS=${CPS:-"0"}
DIR=${DIR:-/dev/shm/egad_ycsb_cr}

# YCSB-F vs YCSB-i canonical recovery configs.
if [ "$WL" = "ycsbi" ]; then
    EPOCHS=${EPOCHS:-10}
    ARGS=(-b ycsbi -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false
          -n 8000000 -N 1000000 -x gpu -e "$EPOCHS" -y hybrid_staging -z "$Z")
else
    EPOCHS=${EPOCHS:-20}
    ARGS=(-b ycsbf -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false
          -n 20000000 -x gpu -e "$EPOCHS" -y hybrid_staging -z "$Z")
fi

NUMA=(numactl --physcpubind=12-23,36-47 --membind=1)
BASEENV=(EPIC_WORKLOAD_AWARE_AUTOSIZER=1
         OMP_DYNAMIC=false
         CUDA_VISIBLE_DEVICES=$GPU EPIC_YCSB_SEED=$SEED)
valonly () { grep -aoE 'STATE-HASH-VALONLY.*0x[0-9a-f]+' "$1" | grep -oE '0x[0-9a-f]+' | tail -1; }
# positional [STATE-HASH] (byte-exact). Recovery reproduces it byte-
# identically, so gating on it too catches value-swaps the value-multiset misses.
pos () { grep -aoE 'STATE-HASH\] .*0x[0-9a-f]+' "$1" | grep -oE '0x[0-9a-f]+' | tail -1; }

# Auto-compute the no-crash DURABLE baseline for this (WL,Z,SEED) if not supplied.
# Durable-on vs off must match (OFF byte-identical), so this also doubles as the
# OFF check when compared to a normal (non-durable) run's VALONLY.
if [ -z "$BASELINE" ]; then
    rm -rf "$DIR"; mkdir -p "$DIR"
    blog=/tmp/ycr_baseline_${WL}_z${Z}_s${SEED}.log
    env "${BASEENV[@]}" EPIC_DURABLE_STORE="$DIR" "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "$blog" 2>&1
    BASELINE=$(valonly "$blog")
    BASEPOS=$(pos "$blog")
    echo "[baseline] no-crash DURABLE $WL z=$Z seed=$SEED -> VALONLY=${BASELINE:-<none>} POS=${BASEPOS:-<none>}"
    # Also a non-durable run to confirm OFF byte-identical.
    nlog=/tmp/ycr_nodurable_${WL}_z${Z}_s${SEED}.log
    env "${BASEENV[@]}" "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "$nlog" 2>&1
    NVAL=$(valonly "$nlog")
    if [ "$NVAL" = "$BASELINE" ] && [ -n "$NVAL" ]; then
        echo "[OFF-check] non-durable VALONLY=$NVAL == durable baseline  (byte-identical OK)"
    else
        echo "[OFF-check] WARNING non-durable VALONLY=${NVAL:-<none>} != durable $BASELINE"
    fi
    rm -rf "$DIR"
fi

pass=0; fail=0; failed=""
echo "YCSB crash-recover sweep: GPU=$GPU WL=$WL z=$Z seed=$SEED epochs=$EPOCHS baseline=$BASELINE CES={$CES} CPS={$CPS}"
echo "----------------------------------------------------------------------"
for ce in $CES; do for cp in $CPS; do
    rm -rf "$DIR"; mkdir -p "$DIR"
    wlog=/tmp/ycr_w_${WL}_e${ce}_p${cp}.log
    rlog=/tmp/ycr_r_${WL}_e${ce}_p${cp}.log
    # worker: durable store + inject real fault at (ce, cp). Expected to die non-zero.
    env "${BASEENV[@]}" EPIC_DURABLE_STORE="$DIR" EPIC_CRASH_AT_EPOCH="$ce" EPIC_CRASH_AT_PHASE="$cp" \
        "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "$wlog" 2>&1
    wrc=$?
    # recover: fresh process re-maps the durable store and replays E-1, E.
    env "${BASEENV[@]}" EPIC_RECOVER_FROM="$DIR" \
        "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "$rlog" 2>&1
    rrc=$?
    rvo=$(valonly "$rlog"); rpos=$(pos "$rlog")
    vfail=$(grep -aE '\[VERIFY\].*FAILED' "$rlog")
    demoted=$(grep -aoE 'demoted [0-9]+ slots' "$rlog" | grep -oE '[0-9]+' | head -1)
    ok=1; why=""
    { [ "$rvo" = "$BASELINE" ] && [ -n "$rvo" ]; } || { ok=0; why+=" VALONLY(${rvo:-none}!=$BASELINE)"; }
    if [ -n "$BASEPOS" ]; then { [ "$rpos" = "$BASEPOS" ] && [ -n "$rpos" ]; } || { ok=0; why+=" POS(${rpos:-none}!=$BASEPOS)"; }; fi
    [ -n "$vfail" ] && { ok=0; why+=" [VERIFY]FAILED"; }
    if [ "$ok" = 1 ]; then
        echo "  PASS  crash@e${ce} p${cp}  worker_rc=$wrc recover_rc=$rrc demoted=${demoted:-?}  VALONLY=$rvo POS=$rpos"
        pass=$((pass+1))
    else
        echo "  FAIL  crash@e${ce} p${cp}  worker_rc=$wrc recover_rc=$rrc diverged:$why  (logs: $wlog $rlog)"
        fail=$((fail+1)); failed+=" e${ce}p${cp}"
    fi
    rm -rf "$DIR"
done; done
echo "----------------------------------------------------------------------"
echo "RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "ALL CRASH-RECOVER CELLS MATCH BASELINE (VALONLY)" || echo "FAILED:$failed"
