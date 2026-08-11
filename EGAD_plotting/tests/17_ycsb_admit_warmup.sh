#!/usr/bin/env bash
# 17_ycsb_admit_warmup: per-epoch admission volume across the run (figure 17).
# Shares data with plot 05 (beyond_hbm); no extra benchmark runs. This stub
# verifies the source logs exist and the plot reads beyond_hbm/ directly.

set -uo pipefail

SRC="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/beyond_hbm"

if [[ ! -d "$SRC" ]]; then
    echo "    missing dir: $SRC  (run 05_beyond_hbm.sh first)"
    exit 1
fi

n=$(ls "$SRC"/ycsbf_120B_hybrid_staging_skew*_rep*.log 2>/dev/null | wc -l)
echo "    ycsb_admit_warmup: $n source logs available."
