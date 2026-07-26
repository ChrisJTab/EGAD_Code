#!/usr/bin/env bash
# 14_breakdown_combined: combined per-phase epoch breakdown across both
# YCSB (4 workloads) and TPCC (tpccdeck W=128). One figure with 10
# horizontal bars (5 workloads x async/sync) that subsumes plots 08 + 10.
#
# Shares data with plots 08 (writeback_breakdown/) and 10
# (tpcc_writeback_breakdown/). No extra benchmark runs needed; this
# stub verifies the source logs exist and the plot script reads from
# both directories directly.

set -uo pipefail

SRC_YCSB="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/writeback_breakdown"
SRC_TPCC="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/tpcc_writeback_breakdown"

missing=0
[[ -d "$SRC_YCSB" ]] || { echo "    missing: $SRC_YCSB  (run 08_writeback_breakdown.sh first)"; missing=$((missing+1)); }
[[ -d "$SRC_TPCC" ]] || { echo "    missing: $SRC_TPCC  (run 10_tpcc_writeback_breakdown.sh first)"; missing=$((missing+1)); }
if [[ $missing -gt 0 ]]; then
    exit 1
fi

n_ycsb=$(ls "$SRC_YCSB"/*.log 2>/dev/null | wc -l)
n_tpcc=$(ls "$SRC_TPCC"/*.log 2>/dev/null | wc -l)
echo "    breakdown_combined: $n_ycsb YCSB + $n_tpcc TPCC source logs available."
