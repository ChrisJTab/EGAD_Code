#!/usr/bin/env bash
# 10_tpcc_writeback_breakdown: TPCC analog of plot 08. Per-phase epoch
# decomposition at W=128 tpccdeck, async vs sync writeback. Two stacked
# horizontal bars showing how async hides the per-table flush across all
# 8 stagers behind the next epoch's prepareEpoch.
#
# Per-stager prepareEpoch wallclocks are aggregated to a single "stage"
# block per epoch (max across stagers, since they overlap on per-stager
# streams under EPIC_PREPARE_EPOCH_PARALLEL). The async writeback
# component is what shrinks vs sync.
#
# 1 W × 2 modes × 3 reps × e=50 = 6 cells. ~3 min.

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/tpcc_writeback_breakdown"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
BINARY="$EPIC_DIR/build/epic_driver"

W=128
REPS=(1 2 3)
EPOCHS=50

run_one() {
    local mode=$1 rep=$2
    local mode_tag="async"
    [[ "$mode" == "false" ]] && mode_tag="sync"
    local log="$OUTDIR/tpccdeck_w${W}_${mode_tag}_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    echo "    [run] $(basename "$log") (-z $mode)"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1 \
    EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1 \
    EPIC_PREPARE_EPOCH_PARALLEL=1 EPIC_WORKLOAD_AWARE_AUTOSIZER=1 \
    OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false \
    CUDA_VISIBLE_DEVICES=0 \
    numactl --physcpubind=0-11,24-35 --membind=0 \
      "$BINARY" -b tpccdeck -d epic -w "$W" -a 0.0 -r false -c 24 \
        -s 100000 -f false -m false -n 20000000 \
        -x gpu -e "$EPOCHS" -y hybrid_staging -z "$mode" \
        > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL ($rc): $log"
        return $rc
    fi
    sleep 1
}

for mode in true false; do
    for rep in "${REPS[@]}"; do
        run_one "$mode" "$rep" || exit $?
    done
done

echo "    tpcc_writeback_breakdown: $(( 2 * ${#REPS[@]} )) cells complete (or skipped)."
