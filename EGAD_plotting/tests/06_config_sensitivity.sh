#!/usr/bin/env bash
# 06_config_sensitivity: hybrid vs cpu_only at three (read, field-split)
# combinations, single workload, single record size, matched ~33.6%
# cache/DB ratio.
#
# The three configs:
#   -rT-fF  full-record read, no field-split           (headline; plot 03/05 config)
#   -rT-fT  full-record read, field-split storage      (falls short: 10x read amplification)
#   -rF-fT  single-field read, field-split storage     (best case for FS)
#
# Cache pressure is held identical via caps:
#   -rT-fF natural at 120B+20M is 100% (no eviction); cap to 6720000 records (33.6%).
#   -rT-fT autosizes to ~67M field-units (33.6% of 200M field-units) - no cap needed.
#   -rF-fT autosizes the same way - no cap.
#
# cpu_only baselines run at the matching read/split config (no cache, no cap).
#
# 1 workload x 1 size x 3 configs x 2 modes x 6 skews x 3 reps = 108 runs.
# Per-cell skip if log already exists unless FORCE_RERUN=1.
#
# Outputs $EGAD_LOGS_DIR/config_sensitivity/<workload>_<config>_<mode>_skew<skew>_rep<rep>.log.

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/config_sensitivity"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
BINARY="$EPIC_DIR/build-small/epic_driver"          # 120 B, ON: hybrid_staging
BINARY_OFF="$EPIC_DIR/build-small-off/epic_driver"  # 120 B, OFF: cpu_only baseline (value-incorrect on ON; RUNNING_TESTS.md §1)

WORKLOAD=ycsbf
SKEWS=(0.01 0.2 0.4 0.6 0.8 0.99)
MODES=(hybrid_staging cpu_only)
REPS=(1 2 3)

# Per-config (r, f) flag pairs.
declare -A R_FLAG=([rTfF]=true  [rTfT]=true  [rFfT]=false)
declare -A F_FLAG=([rTfF]=false [rTfT]=true  [rFfT]=true)
CONFIGS=(rTfF rTfT rFfT)

# Cap value for -rT-fF hybrid only (records, 33.6 % of 20 M).
CAP_RTFF=6720000

run_one() {
    local config=$1 mode=$2 skew=$3 rep=$4
    local log="$OUTDIR/${WORKLOAD}_${config}_${mode}_skew${skew}_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    local r="${R_FLAG[$config]}" f="${F_FLAG[$config]}"
    local epochs=300 cthreads=32 exec_dev="gpu"
    if [[ "$mode" != "hybrid_staging" ]]; then epochs=15; fi  # cpu_only: avg window e3-10 needs >=11 epochs
    if [[ "$mode" == "cpu_only" ]]; then cthreads=24; exec_dev="cpu"; fi

    echo "    [run] $(basename "$log") (e=$epochs, c=$cthreads, r=$r, f=$f)"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    local args="-b $WORKLOAD -d epic -w 1 -a $skew -r $r -c $cthreads -s 100000 \
                -f $f -m false -n 20000000 -x $exec_dev -e $epochs -y $mode -z true"

    local cap_env=""
    if [[ "$config" == "rTfF" && "$mode" == "hybrid_staging" ]]; then
        cap_env="EPIC_YCSB_CACHE_CAP=$CAP_RTFF"
    fi

    if [[ "$mode" == "cpu_only" ]]; then
        env $cap_env CUDA_VISIBLE_DEVICES=0 "$BINARY_OFF" $args > "$log" 2>&1
    else
        env $cap_env CUDA_VISIBLE_DEVICES=0 OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
            numactl --physcpubind=0-11,24-35 --membind=0 \
            "$BINARY" $args > "$log" 2>&1
    fi
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL ($rc): $log"
        return $rc
    fi
    sleep 1
}

for config in "${CONFIGS[@]}"; do
    for mode in "${MODES[@]}"; do
        for skew in "${SKEWS[@]}"; do
            for rep in "${REPS[@]}"; do
                run_one "$config" "$mode" "$skew" "$rep" || exit $?
            done
        done
    done
done

echo "    config_sensitivity: 108 cells complete (or skipped). logs in $OUTDIR"
