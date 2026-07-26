#!/usr/bin/env bash
# 02_ycsb_overlap: YCSB-F rF-fT 120 B (small-records, hybrid),
# 300 epochs so we have a steady-state eviction-path epoch around
# e250-280 to render the overlap plot from.
#
# Outputs $EGAD_LOGS_DIR/ycsb_overlap.log. Skipped if the log already
# exists unless FORCE_RERUN=1.
set -uo pipefail

LOG="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/ycsb_overlap.log"

if [[ -f "$LOG" && "${FORCE_RERUN:-0}" != "1" ]]; then
    echo "    log already exists, skipping (set FORCE_RERUN=1 to override): $LOG"
    exit 0
fi

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
cd "$EPIC_DIR"

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

CUDA_VISIBLE_DEVICES=0 \
OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
numactl --physcpubind=0-11,24-35 --membind=0 \
  ./build-small/epic_driver \
  -b ycsbf -d epic -w 1 -a 0.5 \
  -r false -c 32 -s 100000 \
  -f true  -m false -n 20000000 \
  -x gpu   -e 300 -y hybrid_staging -z true \
  > "$LOG" 2>&1
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "    epic_driver failed (rc=$rc); see $LOG"
    exit $rc
fi
echo "    log written: $LOG"
