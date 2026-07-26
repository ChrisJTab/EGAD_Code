#!/usr/bin/env bash
# 04_three_way: staging overhead when staging isn't strictly needed.
# DB fits HBM (5M records at 120B, -rT-fF), so gpu_only runs natively
# and hybrid_staging has fully-resident cache (no eviction). The question
# this answers is "what does the staging machinery cost when not needed".
# No cpu_only: at this DB size, no one would pick cpu_only over gpu_only.
#
# 4 workloads x 2 modes x 6 skews x 3 reps = 144 runs.
# Per-cell skip if log already exists unless FORCE_RERUN=1.
#
# Outputs $EGAD_LOGS_DIR/three_way/<workload>_<mode>_skew<skew>_rep<rep>.log.

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/three_way"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
BINARY="$EPIC_DIR/build-small/epic_driver"          # 120 B, ON: hybrid_staging
BINARY_OFF="$EPIC_DIR/build-small-off/epic_driver"  # 120 B, OFF: gpu_only baseline (stock layout; ON padding inflates its HBM footprint)

WORKLOADS=(ycsba ycsbb ycsbc ycsbf)
SKEWS=(0.01 0.2 0.4 0.6 0.8 0.99)
MODES=(gpu_only hybrid_staging)
REPS=(1 2 3)

# DB sized to fit HBM (gpu_only can run): 5M records x 256 B = 1.3 GB.
NRECS=5000000

run_one() {
    local bench=$1 mode=$2 skew=$3 rep=$4
    local log="$OUTDIR/${bench}_${mode}_skew${skew}_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    local epochs=300 cthreads=32 exec_dev="gpu"
    if [[ "$mode" != "hybrid_staging" ]]; then epochs=5; fi

    echo "    [run] $(basename "$log") (e=$epochs, c=$cthreads, x=$exec_dev)"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    local args="-b $bench -d epic -w 1 -a $skew -r true -c $cthreads -s 100000 \
                -f false -m false -n $NRECS -x $exec_dev -e $epochs -y $mode -z true"

    local binary="$BINARY"
    [[ "$mode" != "hybrid_staging" ]] && binary="$BINARY_OFF"
    CUDA_VISIBLE_DEVICES=0 OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
        numactl --physcpubind=0-11,24-35 --membind=0 \
        "$binary" $args > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL ($rc): $log"
        return $rc
    fi
    sleep 1
}

for bench in "${WORKLOADS[@]}"; do
    for mode in "${MODES[@]}"; do
        for skew in "${SKEWS[@]}"; do
            for rep in "${REPS[@]}"; do
                run_one "$bench" "$mode" "$skew" "$rep" || exit $?
            done
        done
    done
done

echo "    three_way: 144 cells complete (or skipped). logs in $OUTDIR"
