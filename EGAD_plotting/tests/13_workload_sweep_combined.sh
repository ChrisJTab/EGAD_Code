#!/usr/bin/env bash
# 13_workload_sweep_combined: combined 2x4 panel figure showing the YCSB
# cross-workload skew sweep across BOTH regimes:
#   Top row    (fits in HBM):    hybrid_staging vs gpu_only, 5 M records
#   Bottom row (exceeds HBM):    hybrid_staging vs cpu_only, 20 M records
#
# Shares data with plots 04 (three_way) and 05 (beyond_hbm). No extra
# benchmark runs needed; this stub verifies the source logs exist and
# the plot script reads from beyond_hbm/ and three_way/ directly.

set -uo pipefail

SRC_TW="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/three_way"
SRC_BH="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/beyond_hbm"

missing=0
if [[ ! -d "$SRC_TW" ]]; then
    echo "    missing dir: $SRC_TW  (run 04_three_way.sh first)"
    missing=$((missing + 1))
fi
if [[ ! -d "$SRC_BH" ]]; then
    echo "    missing dir: $SRC_BH  (run 05_beyond_hbm.sh first)"
    missing=$((missing + 1))
fi
if [[ $missing -gt 0 ]]; then
    exit 1
fi

n_tw=$(ls "$SRC_TW"/ycsb*.log 2>/dev/null | wc -l)
n_bh=$(ls "$SRC_BH"/ycsb*.log 2>/dev/null | wc -l)
echo "    workload_sweep_combined: $n_tw three_way + $n_bh beyond_hbm source logs available."
