#!/usr/bin/env bash
# 11_tpcc_cliff: TPCC deep-E cliff demonstration. Long-running e=400 at
# W=128 with --hybrid_hbm_reserve_gb=6 (deliberately undersized cache
# so OL eviction pressure rises with epoch). Two mixes side-by-side:
#
#   tpccdeck: NewOrder rate == Delivery rate, OL backlog stays bounded
#             at ~9000W = 1.15M rows. Throughput stays flat through e=400.
#
#   tpccfull: spec mix 45/43/4/4/4. NewOrder rate > Delivery rate, so
#             the undelivered queue grows monotonically. After ~e133
#             the queue exceeds the cache, eviction starts hitting
#             still-needed slots, and per-epoch throughput cliffs from
#             ~6.5 MTxn/s toward cpu_only's ~5.0 floor.
#
# Plus a cpu_only baseline cell at W=128 tpccdeck for the floor line
# (cpu_only's flush-free design isn't affected by HBM pressure).
#
# 2 mixes x 1 W x 1 mode x 3 reps + 1 cpu_only baseline = 7 cells
# at e=400, ~3 min/cell hybrid + 1.5 min cpu_only ≈ ~25 min total.

set -uo pipefail

# Deep e=400 runs need THP defrag=defer: under the 'madvise' reboot default,
# growing-store huge-page faults direct-compact once node 0's free 2 MB
# blocks fragment away (~e285), stalling the flush worker (deck steps
# 8.3 -> 2.9 MTxn/s). run_all.sh sets this in its preflight; self-apply here
# too so standalone invocations of this test are protected. e<=50 tests are
# insensitive to the setting.
defrag="$(grep -oP '\[\K[^]]+' /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo defer)"
if [[ "$defrag" != "defer" ]]; then
    echo "    defrag is '${defrag}'; setting 'defer' (deep-run requirement)"
    echo defer | sudo -n tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null 2>&1 || true
    defrag="$(grep -oP '\[\K[^]]+' /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || echo defer)"
    if [[ "$defrag" != "defer" ]]; then
        echo "!!! could not set transparent_hugepage/defrag=defer (needs passwordless sudo):"
        echo "    echo defer | sudo tee /sys/kernel/mm/transparent_hugepage/defrag"
        exit 1
    fi
fi

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/tpcc_cliff"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
BINARY="$EPIC_DIR/build/epic_driver"

W=128
REPS=(1 2 3)
EPOCHS=400

run_hybrid() {
    local mix=$1 rep=$2
    local log="$OUTDIR/${mix}_w${W}_hybrid_reserve6_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    echo "    [run] $(basename "$log")"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    # EPIC_NO_HUGE_GROWING=1: this deep-run demo isolates the GPU-cache
    # capacity dynamic, so the growing-store huge-page allocation policy is
    # held at 4 KB. With the madvise active, tpccfull's decline onset moves
    # from ~e300 to ~e120 (reproduced under defrag=defer and with
    # numa_balancing off; mechanism beyond that undiagnosed) — an orthogonal
    # host-allocation effect that would contaminate the cache-pressure story
    # this figure exists to show. e<=50 figures keep the madvise (default).
    EPIC_NO_HUGE_GROWING=1 \
    EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1 \
    EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1 \
    EPIC_PREPARE_EPOCH_PARALLEL=1 EPIC_WORKLOAD_AWARE_AUTOSIZER=1 \
    OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false \
    CUDA_VISIBLE_DEVICES=0 \
    numactl --physcpubind=0-11,24-35 --membind=0 \
      "$BINARY" -b "$mix" -d epic -w "$W" -a 0.0 -r false -c 24 \
        -s 100000 -f false -m false -n 20000000 \
        -x gpu -e "$EPOCHS" -y hybrid_staging -z true \
        --hybrid_hbm_reserve_gb=6 \
        > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL ($rc): $log"
        return $rc
    fi
    sleep 1
}

run_cpu_floor() {
    local rep=$1
    local log="$OUTDIR/tpccdeck_w${W}_cpu_only_floor_rep${rep}.log"

    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi

    echo "    [run] $(basename "$log")"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true

    # cpu_only uses e=50 (not the hybrid's e=400). cpu_only is unaffected
    # by HBM cache pressure (no cache to thrash), so its steady-state
    # number is a constant we just need ONE measurement of. At W=128
    # e=400 cpu_only OOMs the GPU because the OL primary store allocation
    # scales with epochs (insert-pool worst case = num_txns * epochs * 15).
    # cpu_only must run the OFF (stock layout) binary: it is value-incorrect
    # on the ON layout (RUNNING_TESTS.md §1).
    CUDA_VISIBLE_DEVICES=0 \
      "$EPIC_DIR/build-off/epic_driver" -b tpccdeck -d epic -w "$W" -a 0.0 -r false -c 24 \
        -s 100000 -f false -m false -n 20000000 \
        -x cpu -e 50 -y cpu_only \
        > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL ($rc): $log"
        return $rc
    fi
    sleep 1
}

for mix in tpccdeck tpccfull; do
    for rep in "${REPS[@]}"; do
        run_hybrid "$mix" "$rep" || exit $?
    done
done

# Single rep of cpu_only as the floor reference (cpu_only is deterministic
# enough that one rep suffices; cliff-plot only renders its e30-50 median
# as a horizontal line).
run_cpu_floor 1 || exit $?

echo "    tpcc_cliff: $(( 2 * ${#REPS[@]} + 1 )) cells complete (or skipped)."
