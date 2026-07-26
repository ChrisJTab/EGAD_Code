#!/usr/bin/env bash
# 01_tpcc_overlap: canonical TPCC tpccdeck W=128, 50 epochs
# (sufficient for a steady-state eviction-path epoch around e45-e50).
#
# Outputs $EGAD_LOGS_DIR/tpcc_overlap.log. Skipped if the log already
# exists unless FORCE_RERUN=1.
set -uo pipefail

LOG="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/tpcc_overlap.log"

if [[ -f "$LOG" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "    log already exists, skipping (set FORCE_RERUN=1 to override): $LOG"
    exit 0
fi

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
cd "$EPIC_DIR"

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1 \
EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1 \
EPIC_PREPARE_EPOCH_PARALLEL=1 EPIC_WORKLOAD_AWARE_AUTOSIZER=1 \
OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false \
CUDA_VISIBLE_DEVICES=0 numactl --physcpubind=0-11,24-35 --membind=0 \
  ./build/epic_driver -b tpccdeck -d epic -w 128 -a 0.0 -r false -c 24 \
    -s 100000 -f false -m false -n 20000000 -x gpu -e 50 \
    -y hybrid_staging -z true --hybrid_hbm_reserve_gb=6 \
    > "$LOG" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "    epic_driver failed (rc=$rc); see $LOG"
    exit $rc
fi
echo "    log written: $LOG"
