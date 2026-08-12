#!/usr/bin/env bash
# 18_epoch_timeline: measured epoch timeline, async vs sync (figure 18).
# Shares data with plot 08/14 (writeback_breakdown); no extra benchmark
# runs and no extra instrumentation. Every bar comes from log lines the
# driver already prints (per-phase times, transfer_versions, worker_phases).
# This stub verifies the source logs exist.

set -uo pipefail

SRC="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/writeback_breakdown"

if [[ ! -d "$SRC" ]]; then
    echo "    missing dir: $SRC  (run 08_writeback_breakdown.sh first)"
    exit 1
fi

for f in ycsbf_async_rep1.log ycsbf_sync_rep1.log; do
    if [[ ! -f "$SRC/$f" ]]; then
        echo "    missing log: $SRC/$f  (run 08_writeback_breakdown.sh first)"
        exit 1
    fi
done

n=$(ls "$SRC"/ycsbf_{async,sync}_rep*.log 2>/dev/null | wc -l)
echo "    epoch_timeline: $n source logs available."
