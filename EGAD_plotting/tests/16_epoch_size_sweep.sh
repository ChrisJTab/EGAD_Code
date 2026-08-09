#!/usr/bin/env bash
# 16_epoch_size_sweep: epoch-size (batch-size) sensitivity, three systems.
# Sweep the per-epoch transaction count S with everything else held fixed
# and measure (a) throughput vs S and (b) throughput vs the latency each S
# implies (1.5x the epoch wallclock, Epic's convention). Two workloads:
#   - YCSB-F, theta=0.5, 20 M x 120 B records:  S in {5K,10K,25K,50K,100K,200K,400K}
#   - TPC-C deck, W=64:                         S in {5K,25K,100K,400K}
# Three arms per workload:
#   - EGAD (hybrid_staging): build-small (YCSB) / build (TPC-C), the
#     canonical invocations of tests 05 and 09.
#   - EPIC-CPU / EPIC-GPU on YCSB: TRUE UPSTREAM Epic, the epic_stock
#     120 B checkout (./setup_epic_stock.sh --small-records && cmake build;
#     see the requirements check below). Upstream selects modes with -x
#     alone and takes the same -s/-e flags.
#   - EPIC-CPU / EPIC-GPU on TPC-C: this repo's OFF-layout build
#     (build-off), the fig-09 baseline convention. Upstream has no
#     tpccdeck mix, and stock-vs-fork TPC-C CPU parity is 0.99-1.00x
#     (PLOTS.md provenance), so the OFF binary stands in for stock here.
#
# Design rules that make cells comparable across S:
#   - Matched total transactions per run (epochs = total/S), so table
#     growth and generation state in the measured window are identical at
#     every S: EGAD YCSB 35.2 M (long cache warmup is real), stock YCSB
#     8 M, TPC-C 9.6 M. Steady-state windows are the LAST 5.28 M / 5 M /
#     4.8 M transactions respectively (the plot recomputes them from S).
#   - Cache capacity pinned independent of S. YCSB: EPIC_YCSB_CACHE_CAP
#     fixed at the canonical 33.6 % (the autosizer's input subtracts the
#     epoch-proportional version arrays, so an unpinned capacity would
#     covary with S). TPC-C: one unpinned probe run at S=100K reads the
#     workload-aware autosizer's chosen caps, and those exact values are
#     pinned on every TPC-C EGAD cell via the --cache_capacity_* flags.
#   - One discarded warm-up run per binary before its cells (first runs
#     after a fresh build measure ~5 % slow).
# 3 reps per cell; failed runs are renamed <log>.fail.<rc> so a re-run
# retries exactly those. gpu_only-style OOM at large S x total is treated
# as a fit boundary (rep1 failure skips that size's remaining reps).
# (7*3 + 4*3) x 3 reps = 99 cells; roughly an hour end to end.

set -uo pipefail

OUTDIR="${EGAD_LOGS_DIR:?EGAD_LOGS_DIR must be set by run_all.sh}/epoch_size_sweep"
mkdir -p "$OUTDIR"

EPIC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../EGAD" && pwd)"
ROOT="$(cd "$EPIC_DIR/.." && pwd)"
STOCK_DIR="${EGAD_STOCK_DIR:-$ROOT/epic_stock}"

BIN_EGAD_YCSB="$EPIC_DIR/build-small/epic_driver"   # ON layout, 120 B YCSB
BIN_EGAD_TPCC="$EPIC_DIR/build/epic_driver"         # ON layout (TPC-C sizes are spec-fixed)
BIN_OFF="$EPIC_DIR/build-off/epic_driver"           # OFF layout: TPC-C baselines
BIN_STOCK="$STOCK_DIR/build/epic_driver"            # upstream, 120 B YCSB build

for b in "$BIN_EGAD_YCSB" "$BIN_EGAD_TPCC" "$BIN_OFF"; do
    if [[ ! -x "$b" ]]; then
        echo "    missing binary: $b (see build_binaries.sh / RUNNING_TESTS.md section 1)"
        exit 1
    fi
done

STOCK_OK=1
if [[ ! -x "$BIN_STOCK" ]]; then
    STOCK_OK=0
elif ! grep -q 'data\[10\]\[12\]' "$STOCK_DIR/benchmarks/ycsb_table.h" 2>/dev/null; then
    echo "    epic_stock at $STOCK_DIR is the 1 KB variant; this sweep needs --small-records"
    STOCK_OK=0
fi
if [[ $STOCK_OK -eq 0 ]]; then
    echo "    !!! stock 120 B binary not usable at $BIN_STOCK"
    echo "        ./setup_epic_stock.sh --small-records   (or point EGAD_STOCK_DIR at one)"
    echo "        cmake -S \$dest -B \$dest/build -DCMAKE_BUILD_TYPE=Release && cmake --build \$dest/build -j"
    echo "        SKIPPING the YCSB EPIC-CPU / EPIC-GPU arms; EGAD + TPC-C arms still run."
fi

SIZES_YCSB=(5000 10000 25000 50000 100000 200000 400000)
SIZES_TPCC=(5000 25000 100000 400000)
REPS=(1 2 3)

TOTAL_EGAD_YCSB=35200000
TOTAL_STOCK_YCSB=8000000
TOTAL_TPCC=9600000

CACHE_CAP=6720000    # 33.6 % of 20 M, the canonical 120 B ratio (PLOTS.md Appendix B)
TIMEOUT=1800

epochs_for() { echo $(( ($1 + $2 - 1) / $2 )); }

run_cmd() {
    local log=$1; shift
    if [[ -f "$log" && "${FORCE_RERUN:-0}" != "1" ]]; then
        echo "    skip (exists): $(basename "$log")"
        return 0
    fi
    echo "    [run] $(basename "$log")"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true
    timeout $TIMEOUT "$@" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "    FAIL rc=$rc: $(basename "$log")"
        mv "$log" "$log.fail.$rc"
        return $rc
    fi
    sleep 1
}

# ---------------- YCSB ----------------

egad_ycsb() {  # size rep
    local s=$1 rep=$2 e; e=$(epochs_for $TOTAL_EGAD_YCSB "$s")
    run_cmd "$OUTDIR/egad_ycsbf_s${s}_rep${rep}.log" \
      env EPIC_YCSB_CACHE_CAP=$CACHE_CAP CUDA_VISIBLE_DEVICES=0 OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
      numactl --physcpubind=0-11,24-35 --membind=0 \
      "$BIN_EGAD_YCSB" -b ycsbf -d epic -w 1 -a 0.5 -r true -c 32 -s "$s" -f false -m false \
      -n 20000000 -x gpu -e "$e" -y hybrid_staging -z true
}

stock_ycsb() {  # dev(cpu|gpu) size rep
    local dev=$1 s=$2 rep=$3 e c=24; e=$(epochs_for $TOTAL_STOCK_YCSB "$s")
    [[ $dev == gpu ]] && c=32
    run_cmd "$OUTDIR/stock${dev}_ycsbf_s${s}_rep${rep}.log" \
      env CUDA_VISIBLE_DEVICES=0 \
      "$BIN_STOCK" -b ycsbf -d epic -w 1 -a 0.5 -r true -c $c -s "$s" -f false -m false \
      -n 20000000 -x "$dev" -e "$e"
}

# ---------------- TPC-C ----------------

TPCC_HYBRID_ENV=(EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1
                 EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1
                 EPIC_PREPARE_EPOCH_PARALLEL=1 EPIC_WORKLOAD_AWARE_AUTOSIZER=1
                 OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false
                 CUDA_VISIBLE_DEVICES=0)

CAPS=()

tpcc_probe() {
    # Unpinned probe at S=100K, run at the production epoch count (the
    # autosizer's projections depend on num_txns*(epochs+1)); its chosen
    # caps are then pinned on every TPC-C EGAD cell at every S.
    local log="$OUTDIR/tpcc_cap_probe_s100000.log"
    local e; e=$(epochs_for $TOTAL_TPCC 100000)
    run_cmd "$log" \
      env "${TPCC_HYBRID_ENV[@]}" \
      numactl --physcpubind=0-11,24-35 --membind=0 \
      "$BIN_EGAD_TPCC" -b tpccdeck -d epic -w 64 -a 0.0 -r false -c 24 -s 100000 -f false -m false \
      -n 20000000 -x gpu -e "$e" -y hybrid_staging -z true || return 1
    local no o ol
    no=$(grep -oP 'Final caps: NO=\K[0-9]+' "$log" | head -1)
    o=$(grep -oP  'Final caps: NO=[0-9]+ \([0-9.]+% of [0-9]+\); O=\K[0-9]+' "$log" | head -1)
    ol=$(grep -oP '; OL=\K[0-9]+' "$log" | head -1)
    if [[ -z "$no" || -z "$o" || -z "$ol" ]]; then
        echo "    probe yielded no caps (NO='$no' O='$o' OL='$ol')"; return 1
    fi
    CAPS=(--cache_capacity_new_order "$no" --cache_capacity_order "$o" --cache_capacity_orderline "$ol")
    echo "    probe caps pinned for all TPC-C EGAD cells: NO=$no O=$o OL=$ol"
}

egad_tpcc() {  # size rep
    local s=$1 rep=$2 e; e=$(epochs_for $TOTAL_TPCC "$s")
    run_cmd "$OUTDIR/egad_tpccdeck_s${s}_rep${rep}.log" \
      env "${TPCC_HYBRID_ENV[@]}" \
      numactl --physcpubind=0-11,24-35 --membind=0 \
      "$BIN_EGAD_TPCC" -b tpccdeck -d epic -w 64 -a 0.0 -r false -c 24 -s "$s" -f false -m false \
      -n 20000000 -x gpu -e "$e" -y hybrid_staging -z true "${CAPS[@]}"
}

fork_tpcc_cpu() {  # size rep  (fig-09 convention: no env, no pin, -c 24)
    local s=$1 rep=$2 e; e=$(epochs_for $TOTAL_TPCC "$s")
    run_cmd "$OUTDIR/forkcpu_tpccdeck_s${s}_rep${rep}.log" \
      env CUDA_VISIBLE_DEVICES=0 \
      "$BIN_OFF" -b tpccdeck -d epic -w 64 -a 0.0 -r false -c 24 -s "$s" -f false -m false \
      -n 20000000 -x cpu -e "$e" -y cpu_only
}

fork_tpcc_gpu() {  # size rep  (fig-09 convention: flat-index + sizing envs only)
    local s=$1 rep=$2 e; e=$(epochs_for $TOTAL_TPCC "$s")
    run_cmd "$OUTDIR/forkgpu_tpccdeck_s${s}_rep${rep}.log" \
      env EPIC_FLAT_AUX_INDEX=1 EPIC_FLAT_INDEX_OL=1 EPIC_MIX_REALISTIC_SIZING=1 CUDA_VISIBLE_DEVICES=0 \
      "$BIN_OFF" -b tpccdeck -d epic -w 64 -a 0.0 -r false -c 32 -s "$s" -f false -m false \
      -n 20000000 -x gpu -e "$e" -y gpu_only
}

# Rep runner with a fit guard: device memory scales with total x S in the
# projected TPC-C tables, so a size that OOMs on rep1 is skipped rather
# than retried twice more.
run_reps_guarded() {  # fn size
    local fn=$1 s=$2 rep
    for rep in "${REPS[@]}"; do
        "$fn" "$s" "$rep"
        local rc=$?
        if [[ $rep -eq 1 && $rc -ne 0 ]]; then
            echo "    [guard] $fn s=$s rep1 failed (rc=$rc) -- skipping reps 2,3 for this size"
            return 0
        fi
    done
}

# ---------------- schedule ----------------

echo "    warm-up runs (discarded)"
run_cmd "$OUTDIR/warmup_egad.log" \
  env EPIC_YCSB_CACHE_CAP=$CACHE_CAP CUDA_VISIBLE_DEVICES=0 OMP_DYNAMIC=false OMP_NUM_THREADS=12 \
  numactl --physcpubind=0-11,24-35 --membind=0 \
  "$BIN_EGAD_YCSB" -b ycsbf -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false \
  -n 20000000 -x gpu -e 40 -y hybrid_staging -z true
if [[ $STOCK_OK -eq 1 ]]; then
    run_cmd "$OUTDIR/warmup_stockcpu.log" \
      env CUDA_VISIBLE_DEVICES=0 "$BIN_STOCK" -b ycsbf -d epic -w 1 -a 0.5 -r true -c 24 \
      -s 100000 -f false -m false -n 20000000 -x cpu -e 10
    run_cmd "$OUTDIR/warmup_stockgpu.log" \
      env CUDA_VISIBLE_DEVICES=0 "$BIN_STOCK" -b ycsbf -d epic -w 1 -a 0.5 -r true -c 32 \
      -s 100000 -f false -m false -n 20000000 -x gpu -e 10
fi

echo "    TPC-C cap probe"
tpcc_probe || echo "    (TPC-C EGAD cells will be skipped without caps)"

echo "    YCSB EGAD"
for s in "${SIZES_YCSB[@]}"; do
    for rep in "${REPS[@]}"; do egad_ycsb "$s" "$rep"; done
done

if [[ $STOCK_OK -eq 1 ]]; then
    echo "    YCSB stock CPU"
    for s in "${SIZES_YCSB[@]}"; do
        for rep in "${REPS[@]}"; do stock_ycsb cpu "$s" "$rep"; done
    done
    echo "    YCSB stock GPU"
    for s in "${SIZES_YCSB[@]}"; do
        for rep in "${REPS[@]}"; do stock_ycsb gpu "$s" "$rep"; done
    done
fi

echo "    TPC-C EGAD"
if [[ ${#CAPS[@]} -gt 0 ]]; then
    for s in "${SIZES_TPCC[@]}"; do run_reps_guarded egad_tpcc "$s"; done
else
    echo "    skipped (no caps)"
fi

echo "    TPC-C cpu_only"
for s in "${SIZES_TPCC[@]}"; do run_reps_guarded fork_tpcc_cpu "$s"; done

echo "    TPC-C gpu_only"
for s in "${SIZES_TPCC[@]}"; do run_reps_guarded fork_tpcc_gpu "$s"; done

fails=$(ls "$OUTDIR"/*.fail.* 2>/dev/null | wc -l)
echo "    epoch_size_sweep complete; failed runs: $fails"
if [[ $fails -gt 0 ]]; then
    ls "$OUTDIR"/*.fail.* 2>/dev/null
    exit 1
fi
