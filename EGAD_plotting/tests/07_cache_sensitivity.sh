#!/usr/bin/env bash
# 07_cache_sensitivity: hybrid_staging throughput as a function of cache/DB
# ratio, with gpu_only (ceiling) and cpu_only (floor) reference lines.
# Single workload (ycsbf), single record size (120B), single skew (0.5).
#
# The cache sweep:
#   5 %  -> EPIC_YCSB_CACHE_CAP=1000000   (1 M records)
#   10 % ->                     2000000   (2 M)
#   25 % ->                     5000000   (5 M)
#   50 % ->                    10000000   (10 M)
#   75 % ->                    15000000   (15 M)
#   100% -> no cap (autosizer picks 20 M = full DB; no eviction)
#
# gpu_only and cpu_only don't depend on the cache, so they're measured
# once (3 reps) and rendered as horizontal reference lines.
#
# 6 hybrid x 3 reps + 1 gpu_only x 3 reps + 1 cpu_only x 3 reps = 24 cells.
# Per-cell skip if log already exists unless FORCE_RERUN=1.
#
# Outputs $EGAD_LOGS_DIR/cache_sensitivity/<workload>_<mode>_[cache<pct>_]rep<rep>.log.

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/cache_sensitivity"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
BINARY="$EPIC_DIR/build-small/epic_driver"          # 120 B, ON: hybrid_staging
BINARY_OFF="$EPIC_DIR/build-small-off/epic_driver"  # 120 B, OFF: reference lines

WORKLOAD=ycsbf
SKEW=0.5
REPS=(1 2 3)

# Cache ratios (%) -> EPIC_YCSB_CACHE_CAP value in records.
# 100% is the no-cap case (autosizer naturally picks 20M = full DB).
declare -A CAP_VALUE=(
    [5]=1000000
    [10]=2000000
    [25]=5000000
    [50]=10000000
    [75]=15000000
    [100]=0
)
CACHE_RATIOS=(5 10 25 50 75 100)

run_hybrid() {
    local pct=$1 rep=$2
    local log="$OUTDIR/${WORKLOAD}_hybrid_staging_cache${pct}_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    local cap_val="${CAP_VALUE[$pct]}"
    local cap_env=""
    if [[ "$cap_val" != "0" ]]; then
        cap_env="EPIC_YCSB_CACHE_CAP=$cap_val"
    fi

    echo "    [run] $(basename "$log") (cache=${pct}%, cap=${cap_val:-none})"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    local args="-b $WORKLOAD -d epic -w 1 -a $SKEW -r true -c 32 -s 100000 \
                -f false -m false -n 20000000 -x gpu -e 300 -y hybrid_staging -z true"

    env $cap_env CUDA_VISIBLE_DEVICES=0 OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
        numactl --physcpubind=0-11,24-35 --membind=0 \
        "$BINARY" $args > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL ($rc): $log"
        return $rc
    fi
    sleep 1
}

run_baseline() {
    local mode=$1 rep=$2
    local log="$OUTDIR/${WORKLOAD}_${mode}_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    local cthreads=32 exec_dev="gpu"
    local epochs=5
    if [[ "$mode" == "cpu_only" ]]; then cthreads=24; exec_dev="cpu"; epochs=15; fi  # cpu_only: avg window e3-10 needs >=11

    echo "    [run] $(basename "$log") (mode=$mode, e=$epochs)"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    local args="-b $WORKLOAD -d epic -w 1 -a $SKEW -r true -c $cthreads -s 100000 \
                -f false -m false -n 20000000 -x $exec_dev -e $epochs -y $mode -z true"

    # Both reference lines run the OFF (stock layout) binary: cpu_only is
    # value-incorrect on ON, and gpu_only's ON padding inflates its HBM
    # footprint (RUNNING_TESTS.md §1).
    if [[ "$mode" == "cpu_only" ]]; then
        CUDA_VISIBLE_DEVICES=0 "$BINARY_OFF" $args > "$log" 2>&1
    else
        CUDA_VISIBLE_DEVICES=0 OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
            numactl --physcpubind=0-11,24-35 --membind=0 \
            "$BINARY_OFF" $args > "$log" 2>&1
    fi
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL ($rc): $log"
        return $rc
    fi
    sleep 1
}

for pct in "${CACHE_RATIOS[@]}"; do
    for rep in "${REPS[@]}"; do
        run_hybrid "$pct" "$rep" || exit $?
    done
done

for mode in gpu_only cpu_only; do
    for rep in "${REPS[@]}"; do
        run_baseline "$mode" "$rep" || exit $?
    done
done

echo "    cache_sensitivity: 24 cells complete (or skipped). logs in $OUTDIR"
