#!/usr/bin/env bash
# 03_record_size: YCSB-F skew sweep at -rT-fF for 120 B vs 1 KB records,
# hybrid_staging vs cpu_only. The "limits" plot: hybrid wins big at 120 B
# everywhere, and converges to / crosses below cpu_only at 1 KB low skew.
#
# 6 skews x 2 sizes x 2 modes x 3 reps = 72 runs.
# Per-cell skip if its log already exists unless FORCE_RERUN=1.
#
# Outputs $EGAD_LOGS_DIR/record_size/<size>_<mode>_skew<skew>_rep<rep>.log.

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/record_size"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"

SKEWS=(0.01 0.2 0.4 0.6 0.8 0.99)
SIZES=(120B 1KB)
MODES=(hybrid_staging cpu_only)
REPS=(1 2 3)

run_one() {
    local size=$1 mode=$2 skew=$3 rep=$4
    local log="$OUTDIR/${size}_${mode}_skew${skew}_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    local binary epochs cthreads exec_dev
    # hybrid_staging needs the ON (hybrid record layout) build; the cpu_only
    # baseline must run the stock OFF layout (cpu_only is value-incorrect on
    # ON; see RUNNING_TESTS.md §1).
    if [[ "$size" == "120B" ]]; then
        binary="$EPIC_DIR/build-small/epic_driver"
        [[ "$mode" != "hybrid_staging" ]] && binary="$EPIC_DIR/build-small-off/epic_driver"
    else
        binary="$EPIC_DIR/build/epic_driver"
        [[ "$mode" != "hybrid_staging" ]] && binary="$EPIC_DIR/build-off/epic_driver"
    fi
    epochs=300; cthreads=32; exec_dev="gpu"
    if [[ "$mode" != "hybrid_staging" ]]; then epochs=15; fi  # cpu_only: avg window e3-10 needs >=11 epochs
    if [[ "$mode" == "cpu_only" ]]; then cthreads=24; exec_dev="cpu"; fi

    echo "    [run] $(basename "$log") (e=$epochs, c=$cthreads, x=$exec_dev)"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    local args="-b ycsbf -d epic -w 1 -a $skew -r true -c $cthreads -s 100000 \
                -f false -m false -n 20000000 -x $exec_dev -e $epochs -y $mode -z true"

    # Matched-ratio framing: at 1KB+20M+-rT-fF the autosizer naturally picks
    # ~10.54M records (~52.7% cache/DB ratio). At 120B+20M+-rT-fF the
    # autosizer would pick the full 20M (no eviction). To compare per-byte
    # transfer cost without confounding cache pressure, cap the 120B hybrid
    # cache at the 1KB autosizer's pick so both record sizes face the same
    # eviction rate per epoch. cpu_only is unaffected (no cache).
    local cap_env=""
    if [[ "$size" == "120B" && "$mode" == "hybrid_staging" ]]; then
        cap_env="EPIC_YCSB_CACHE_CAP=10544194"
    fi

    if [[ "$mode" == "cpu_only" ]]; then
        env $cap_env CUDA_VISIBLE_DEVICES=0 "$binary" $args > "$log" 2>&1
    else
        env $cap_env CUDA_VISIBLE_DEVICES=0 OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
            numactl --physcpubind=0-11,24-35 --membind=0 \
            "$binary" $args > "$log" 2>&1
    fi
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL ($rc): $log"
        return $rc
    fi
    sleep 1
}

for size in "${SIZES[@]}"; do
    for mode in "${MODES[@]}"; do
        for skew in "${SKEWS[@]}"; do
            for rep in "${REPS[@]}"; do
                run_one "$size" "$mode" "$skew" "$rep" || exit $?
            done
        done
    done
done

echo "    record_size: 72 cells complete (or skipped). logs in $OUTDIR"
