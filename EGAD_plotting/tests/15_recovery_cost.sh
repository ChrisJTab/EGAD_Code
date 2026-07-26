#!/usr/bin/env bash
# 15_recovery_cost: cost of GPU-crash recovery on TPC-C.
# Three modes per warehouse count, same build (epic@e66f6af, the durable-recovery
# path) so the off-vs-on gap is pure recovery overhead:
#   off = non-recovery hybrid_staging (canonical fig-09 config, node 0)
#   on  = recovery: off + EPIC_DURABLE_STORE (durable Primary Store, always
#         provisioned on create -- the ship config; lazy commit is not viable)
#   cpu = stock EPIC cpu_only floor (no env, no NUMA pin, -c 24, OFF binary)
# tpccdeck mix, e=50, steady-state window e30-50 (per feedback_steady_state_windows).
# 6 W x 3 modes x 3 reps = 54 runs. drop_caches before each; durable store in
# /dev/shm (cleaned before+after every recovery run; ~34 GB at W=128 < 63 GB).
# Per-cell skip if log exists unless FORCE_RERUN=1.
#
# Outputs $EGAD_LOGS_DIR/recovery_cost/tpccdeck_<mode>_w<W>_rep<rep>.log

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs}/recovery_cost"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
BIN="$EPIC_DIR/build/epic_driver"          # ON (hybrid layout): off/on hybrid_staging modes
BIN_OFF="$EPIC_DIR/build-off/epic_driver"  # OFF (stock layout): the cpu_only floor
DURDIR="/dev/shm/egad_rec_cost"

WS=(4 8 16 32 64 128)
MODES=(off on cpu)
REPS=(1 2 3)

# Canonical TPC-C hybrid env (PLOTS.md "TPCC run convention"); node-0 pin per the
# Phase-8 GPU0/node0 measurement.
BASEENV=(EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1
         EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1
         EPIC_PREPARE_EPOCH_PARALLEL=1 EPIC_WORKLOAD_AWARE_AUTOSIZER=1
         OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false
         CUDA_VISIBLE_DEVICES=0)
NUMA=(numactl --physcpubind=0-11,24-35 --membind=0)

cleanup_dur() { rm -rf "$DURDIR" 2>/dev/null || true; }
trap cleanup_dur EXIT

run_one() {
    local mode=$1 W=$2 rep=$3
    local log="$OUTDIR/tpccdeck_${mode}_w${W}_rep${rep}.log"
    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"; return 0
    fi
    echo "    [run] $(basename "$log") $(date +%H:%M:%S)"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    local ARGS=(-b tpccdeck -d epic -w "$W" -a 0.0 -r false -c 24 -s 100000
                -f false -m false -n 20000000 -x gpu -e 50 -y hybrid_staging -z true)

    if [[ "$mode" == "cpu" ]]; then
        # stock EPIC cpu_only: no env, no NUMA pin, CPU executor, OFF binary
        CUDA_VISIBLE_DEVICES=0 "$BIN_OFF" \
            -b tpccdeck -d epic -w "$W" -a 0.0 -r false -c 24 -s 100000 \
            -f false -m false -n 20000000 -x cpu -e 50 -y cpu_only -z true \
            > "$log" 2>&1
    elif [[ "$mode" == "on" ]]; then
        cleanup_dur; mkdir -p "$DURDIR"
        env "${BASEENV[@]}" EPIC_DURABLE_STORE="$DURDIR" \
            "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "$log" 2>&1
        cleanup_dur
    else  # off
        env "${BASEENV[@]}" "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "$log" 2>&1
    fi
    local rc=$?
    [[ $rc -ne 0 ]] && echo "    FAIL ($rc): $log"
    sleep 1
    return 0
}

echo "GPU tenancy check:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -3 || true
for W in "${WS[@]}"; do
    for mode in "${MODES[@]}"; do
        for rep in "${REPS[@]}"; do
            run_one "$mode" "$W" "$rep"
        done
    done
done
echo "    recovery_cost: 54 cells complete (or skipped). logs in $OUTDIR"
echo "ALL_DONE"
