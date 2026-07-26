#!/usr/bin/env bash
# 12_ycsb_warmup: per-skew per-epoch throughput trajectory at 1 KB records.
# Shares data with plot 03 (record_size/); verifies the 1KB hybrid_staging
# cells exist there. The plot 03 sweep runs ycsbf -rT-fF at 1 KB, 20 M
# records, autosizer's natural cache pick (~52.7 % cache/DB ratio), e=300,
# 3 reps per skew. No extra benchmark runs needed.

set -uo pipefail

SRC_DIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/record_size"
SKEWS=(0.01 0.2 0.4 0.6 0.8 0.99)
REPS=(1 2 3)

missing=0
for skew in "${SKEWS[@]}"; do
    for rep in "${REPS[@]}"; do
        f="$SRC_DIR/1KB_hybrid_staging_skew${skew}_rep${rep}.log"
        if [[ ! -f "$f" ]]; then
            echo "    missing: $f"
            missing=$((missing + 1))
        fi
    done
done
if [[ $missing -gt 0 ]]; then
    echo "    ycsb_warmup: $missing source logs missing under record_size/."
    echo "    Run tests/03_record_size.sh first."
    exit 1
fi
echo "    ycsb_warmup: $(( ${#SKEWS[@]} * ${#REPS[@]} )) source logs available (shared with plot 03)."
