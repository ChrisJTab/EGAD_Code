#!/usr/bin/env bash
# 08_writeback_breakdown: per-phase epoch-time decomposition for hybrid_staging
# in async (-z true) vs sync (-z false) writeback, across the four YCSB
# workloads. Single skew (theta=0.5), 120B records, 20M records, -rT-fF,
# 33.6% cache cap (matched-ratio convention from plots 03/05/06/07).
#
# Each cell logs steady-state epochs e1-e40; the plotter averages across
# e30-e40 for per-phase wallclock medians. The seven phase blocks rendered:
#   index_transfer, indexing, submission, staging, initialization, execution,
#   writeback (= flush in sync; flush_sync_prev + flush_start_async in async)
#
# init_transfer and exec_transfer are consistently 0 us, so they're omitted.
#
# 4 workloads x 2 modes x N reps cells. Start with REPS=(1) for style preview;
# bump to (1 2 3) after style is locked.

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/writeback_breakdown"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
BINARY="$EPIC_DIR/build-small/epic_driver"   # 120 B records

WORKLOADS=(ycsba ycsbb ycsbc ycsbf)
SKEW=0.5
REPS=(1 2 3)

# Match the cache pressure used in plots 05/06/07 (33.6% of 20M records).
CACHE_CAP=6720000

# 40 epochs gets us comfortably past warmup; e30-40 is the steady-state
# window for per-phase median (different from the throughput-headline
# e250-280 because we only need per-phase consistency, not a long average).
EPOCHS=40

run_one() {
    local bench=$1 mode=$2 rep=$3
    local mode_tag="async"
    [[ "$mode" == "false" ]] && mode_tag="sync"
    local log="$OUTDIR/${bench}_${mode_tag}_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    echo "    [run] $(basename "$log") (-z $mode)"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    local args="-b $bench -d epic -w 1 -a $SKEW -r true -c 32 -s 100000 \
                -f false -m false -n 20000000 -x gpu -e $EPOCHS -y hybrid_staging -z $mode"

    EPIC_YCSB_CACHE_CAP=$CACHE_CAP CUDA_VISIBLE_DEVICES=0 \
        OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
        numactl --physcpubind=0-11,24-35 --membind=0 \
        "$BINARY" $args > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL ($rc): $log"
        return $rc
    fi
    sleep 1
}

for bench in "${WORKLOADS[@]}"; do
    for mode in true false; do  # true=async, false=sync
        for rep in "${REPS[@]}"; do
            run_one "$bench" "$mode" "$rep" || exit $?
        done
    done
done

echo "    writeback_breakdown: $(( ${#WORKLOADS[@]} * 2 * ${#REPS[@]} )) cells complete (or skipped)."
