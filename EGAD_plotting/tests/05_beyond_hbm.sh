#!/usr/bin/env bash
# 05_beyond_hbm: cross-workload validation of the headline finding.
# Hybrid vs cpu_only at the headline config (120 B + -rT-fF + 20 M with the
# cache cap that produces ~33.6 % cache/DB ratio (matched-ratio convention,
# Appendix B; plots 05-08 share it) so the cache pressure is real, not "everything fits"
# trivial). Single record size only -- the 120 B vs 1 KB story already lives
# in plot 03; this plot generalizes the headline (hybrid wins over cpu_only)
# across all four YCSB workloads. The -rF-fT / -rT-fT config sensitivity
# story lives in plot 06.
#
# 4 workloads x 1 size x 2 modes x 6 skews x 3 reps = 144 runs.
# Per-cell skip if log already exists unless FORCE_RERUN=1.
#
# Outputs $EGAD_LOGS_DIR/beyond_hbm/<workload>_<size>_<mode>_skew<skew>_rep<rep>.log.

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/beyond_hbm"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"

WORKLOADS=(ycsba ycsbb ycsbc ycsbf)
SIZES=(120B)   # 1KB dropped: covered by plot 03's record-size sensitivity
SKEWS=(0.01 0.2 0.4 0.6 0.8 0.99)
MODES=(hybrid_staging cpu_only)
REPS=(1 2 3)

# Headline cache ratio: 33.6 % cache/DB = 6720000 of 20 M records, matching the
# field-unit autosizer's natural 33.6 % pick at -rT-fT/-rF-fT so plots 05-08 use
# one consistent ratio (Appendix B). Applied as EPIC_YCSB_CACHE_CAP on 120 B hybrid.
CACHE_CAP=6720000

run_one() {
    local bench=$1 size=$2 mode=$3 skew=$4 rep=$5
    local log="$OUTDIR/${bench}_${size}_${mode}_skew${skew}_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    local binary
    # ON build for hybrid_staging; OFF build for the cpu_only baseline
    # (value-incorrect on the ON layout; RUNNING_TESTS.md §1).
    if [[ "$size" == "120B" ]]; then
        binary="$EPIC_DIR/build-small/epic_driver"
        [[ "$mode" != "hybrid_staging" ]] && binary="$EPIC_DIR/build-small-off/epic_driver"
    else
        binary="$EPIC_DIR/build/epic_driver"
        [[ "$mode" != "hybrid_staging" ]] && binary="$EPIC_DIR/build-off/epic_driver"
    fi

    local epochs=300 cthreads=32 exec_dev="gpu"
    if [[ "$mode" != "hybrid_staging" ]]; then epochs=15; fi  # cpu_only: avg window e3-10 needs >=11 epochs
    if [[ "$mode" == "cpu_only" ]]; then cthreads=24; exec_dev="cpu"; fi

    echo "    [run] $(basename "$log") (e=$epochs, c=$cthreads, x=$exec_dev)"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    local args="-b $bench -d epic -w 1 -a $skew -r true -c $cthreads -s 100000 \
                -f false -m false -n 20000000 -x $exec_dev -e $epochs -y $mode -z true"

    # Cache cap only on 120 B hybrid; 1 KB hybrid uses the autosizer's
    # natural ~52.7 % pick, and cpu_only has no cache to cap.
    local cap_env=""
    if [[ "$size" == "120B" && "$mode" == "hybrid_staging" ]]; then
        cap_env="EPIC_YCSB_CACHE_CAP=$CACHE_CAP"
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

for bench in "${WORKLOADS[@]}"; do
    for size in "${SIZES[@]}"; do
        for mode in "${MODES[@]}"; do
            for skew in "${SKEWS[@]}"; do
                for rep in "${REPS[@]}"; do
                    run_one "$bench" "$size" "$mode" "$skew" "$rep" || exit $?
                done
            done
        done
    done
done

echo "    beyond_hbm: 144 cells complete (or skipped). logs in $OUTDIR"
