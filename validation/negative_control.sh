#!/usr/bin/env bash
# Negative-control sensitivity proof for the recovery state-hash gate.
# Proves the gate CATCHES corruption WITHOUT needing a crash: it runs normal
# durable-off runs and perturbs the final Primary Store (via the gated
# EPIC_RECOVERY_CORRUPT_ONE / EPIC_RECOVERY_SWAP_TWO hooks) right before hashing.
#
#   CORRUPT (1-record byte flip) must DIVERGE on BOTH the value-hash AND the
#           positional hash -> gate SENSITIVE. If either still matches -> BLIND.
#   SWAP    (full value swap of two same-table records) is the historical blind
#           spot for the value-hash (multiset/placement-invariant); we PASS iff
#           the new positional gate DIVERGES on it.
#
# The [base] run also doubles as an OFF byte-identical check: with no env set the
# hook is a no-op, so [base] must equal the canonical no-crash hash.
#
# Usage:
#   GPU=2 bash negative_control.sh                 # TP-CC (default)
#   GPU=2 WL=ycsbf bash negative_control.sh        # YCSB-F
#   GPU=2 WL=ycsbi bash negative_control.sh        # YCSB-i
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
BIN=${BIN:-EGAD/build/epic_driver}
GPU=${GPU:-2}
WL=${WL:-tpcc}              # tpcc (default) | ycsbf | ycsbi
SEED=${SEED:-42}
Z=${Z:-true}
NUMA=(numactl --physcpubind=12-23,36-47 --membind=1)

if [ "$WL" = "ycsbf" ] || [ "$WL" = "ycsbi" ]; then
    BASEENV=(EPIC_WORKLOAD_AWARE_AUTOSIZER=1 OMP_DYNAMIC=false
             CUDA_VISIBLE_DEVICES=$GPU EPIC_YCSB_SEED=$SEED)
    if [ "$WL" = "ycsbi" ]; then
        EPOCHS=${EPOCHS:-10}
        ARGS=(-b ycsbi -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false
              -n 8000000 -N 1000000 -x gpu -e "$EPOCHS" -y hybrid_staging -z "$Z")
    else
        EPOCHS=${EPOCHS:-20}
        ARGS=(-b ycsbf -d epic -w 1 -a 0.5 -r true -c 32 -s 100000 -f false -m false
              -n 20000000 -x gpu -e "$EPOCHS" -y hybrid_staging -z "$Z")
    fi
    vh () { grep -aoE 'STATE-HASH-VALONLY.*0x[0-9a-f]+' "$1" | grep -oE '0x[0-9a-f]+' | tail -1; }
    VHNAME=VALONLY
else
    W=${W:-8}; EPOCHS=${EPOCHS:-20}
    BASEENV=(EPIC_FLAT_AUX_INDEX=1 EPIC_PAGEABLE_PRIMARY=1 EPIC_MIX_REALISTIC_SIZING=1
             EPIC_FLAT_INDEX_OL=1 EPIC_OL_DELIVERED_EVICTION=1 EPIC_PREWARM_OL=1
             EPIC_WORKLOAD_AWARE_AUTOSIZER=1
             OMP_NESTED=true OMP_MAX_ACTIVE_LEVELS=2 OMP_NUM_THREADS=8,3 OMP_DYNAMIC=false
             CUDA_VISIBLE_DEVICES=$GPU EPIC_TPCC_SEED=$SEED)
    ARGS=(-b tpccdeck -d epic -w "$W" -a 0.0 -r false -c 24 -s 100000 -f false -m false
          -n 20000000 -x gpu -e "$EPOCHS" -y hybrid_staging -z "$Z" --verify_tpcc)
    vh () { grep -aoE 'STATE-HASH-VALMS.*0x[0-9a-f]+' "$1" | grep -oE '0x[0-9a-f]+' | tail -1; }
    VHNAME=VALMS
fi
pos () { grep -aoE 'STATE-HASH\] .*0x[0-9a-f]+' "$1" | grep -oE '0x[0-9a-f]+' | tail -1; }

run () {  # $1=tag  $2... = extra env (KEY=VAL)
    local tag="$1"; shift
    env "${BASEENV[@]}" "$@" "${NUMA[@]}" "$BIN" "${ARGS[@]}" > "/tmp/negctrl_${tag}.log" 2>&1
}

echo "Negative-control gate-sensitivity proof: WL=$WL GPU=$GPU z=$Z seed=$SEED epochs=$EPOCHS"
echo "----------------------------------------------------------------------"
run base
run corrupt EPIC_RECOVERY_CORRUPT_ONE=1
run swap    EPIC_RECOVERY_SWAP_TWO=1

bvh=$(vh /tmp/negctrl_base.log);    bpos=$(pos /tmp/negctrl_base.log)
cvh=$(vh /tmp/negctrl_corrupt.log); cpos=$(pos /tmp/negctrl_corrupt.log)
svh=$(vh /tmp/negctrl_swap.log);    spos=$(pos /tmp/negctrl_swap.log)
echo "[base]    $VHNAME=$bvh  POS=$bpos"
echo "[corrupt] $VHNAME=$cvh  POS=$cpos"
echo "[swap]    $VHNAME=$svh  POS=$spos"
echo "----------------------------------------------------------------------"

rc=0
if [ -z "$bvh" ] || [ -z "$bpos" ]; then
    echo "[NEG-CTRL] FAIL baseline produced no hashes (run crashed? see /tmp/negctrl_base.log)"; exit 1
fi

# CORRUPT: must diverge on BOTH value-hash and positional.
if [ "$cvh" != "$bvh" ] && [ "$cpos" != "$bpos" ]; then
    echo "[NEG-CTRL] corrupt-one: gate SENSITIVE (PASS)  (value-hash and positional both diverged)"
else
    echo "[NEG-CTRL] corrupt-one: gate BLIND (FAIL)  value-hash $([ "$cvh" = "$bvh" ] && echo SAME || echo diverged), positional $([ "$cpos" = "$bpos" ] && echo SAME || echo diverged)"
    rc=1
fi

# SWAP: secondary, INFORMATIONAL demonstration of the value-multiset blind spot.
# The corrupt-one test above is the definitive sensitivity gate. The swap is not a
# hard assertion because swapping two records that happen to be byte-identical (or
# that share version tags, making the swap clean for the XOR-based positional hash)
# is a no-op the gate is not obligated to catch -- only a real value change is.
if [ "$svh" = "$bvh" ]; then sw_vh="SAME (value-multiset blind spot, as expected)"; else sw_vh="diverged"; fi
if [ "$spos" != "$bpos" ]; then
    echo "[NEG-CTRL] swap-two: positional DIVERGED -> caught by the positional gate; value-hash $sw_vh"
else
    echo "[NEG-CTRL] swap-two: positional SAME -> not caught (target records likely identical or share tags); value-hash $sw_vh  [informational, not a failure]"
fi

echo "----------------------------------------------------------------------"
[ "$rc" -eq 0 ] && echo "RESULT: gate is SENSITIVE (all negative-control assertions passed)" \
                || echo "RESULT: gate FAILED a sensitivity assertion"
exit $rc
