//
// Created by Christian on 2026-05-23.
//
// CPU-side durable shadow of the per-table key->CRID mapping. Owned by
// TpccDb, lifetime independent of TpccGpuIndex. Survives a GPU crash
// because it is plain host memory; serves as the rebuild source for the
// GPU index during recovery.
//
// Mirrors the GPU primary index on inserts: every CRID admitted to the
// GPU also lands here under the same key. Sentinel value 0xffffffffu
// marks an empty slot. Eight flat std::vector<uint32_t> per table, sized
// to the worst-case key universe at construction; the five static tables
// (W/D/C/I/S) and the initial population of the three growing tables
// (NO/O/OL) are filled by loadInitialData(); the per-epoch mirror block
// in TpccGpuIndex::indexTxns extends NO/O/OL via mirrorEpoch().
//
// Used by TpccGpuIndex for rebuildIndexesFromShadow (recovery:
// repopulate the GPU index after a crash).

#ifndef EPIC_BENCHMARKS_TPCC_CPU_SHADOW_INDEX_H
#define EPIC_BENCHMARKS_TPCC_CPU_SHADOW_INDEX_H

#include <cstdint>
#include <vector>

#include <benchmarks/tpcc_config.h>
#include <benchmarks/tpcc_table.h>

namespace epic::tpcc {

// Per-table free-list cursors for the three growing tables. Used by the
// recovery API to communicate "rollback target" / "current state" between
// the recovery driver, the shadow, and the GPU index.
struct TpccFreeStarts
{
    uint32_t new_order  = 0;
    uint32_t order      = 0;
    uint32_t order_line = 0;
};

// === Dense linear encodings of each table's key fields ===
//
// For W / D / C / I / S the encoding is dense over the static key universe
// and dense_idx == initial-population CRID. For NO / O, the per-(w,d)
// stride is a runtime max_o (the largest o_id that will ever appear,
// derived from the configured table size); the stride must NOT be the
// initial 3000, or run-time inserts with o > 3000 alias onto
// initial-population slots. OrderLine uses the same encoding as the GPU flat OL index
// (denseIdxOL, declared in tpcc_flat_index.h) so the shadow and the flat
// array share dense_idx -> slot semantics.
inline uint32_t denseIdxW(uint32_t w)                          { return w - 1u; }
inline uint32_t denseIdxD(uint32_t w, uint32_t d)              { return (w - 1u) * 10u + (d - 1u); }
inline uint32_t denseIdxC(uint32_t w, uint32_t d, uint32_t c)  { return ((w - 1u) * 10u + (d - 1u)) * 3000u + (c - 1u); }
inline uint32_t denseIdxI(uint32_t i)                          { return i - 1u; }
inline uint32_t denseIdxS(uint32_t w, uint32_t i)              { return (w - 1u) * 100000u + (i - 1u); }
inline uint32_t denseIdxNO(uint32_t w, uint32_t d, uint32_t o, uint32_t max_o) { return ((w - 1u) * 10u + (d - 1u)) * max_o + (o - 1u); }
inline uint32_t denseIdxO (uint32_t w, uint32_t d, uint32_t o, uint32_t max_o) { return ((w - 1u) * 10u + (d - 1u)) * max_o + (o - 1u); }

class TpccCpuShadowIndex
{
public:
    // Sentinel marking an empty shadow slot. Matches the GPU-side cuco
    // value sentinel so dense_idx -> sentinel translates the same way on
    // either side of the comparison.
    static constexpr uint32_t kSentinel = 0xffffffffu;

    // maxO_ol is the per-(w,d) OrderLine dense stride. When the GPU
    // flat-OL path is active (EPIC_FLAT_INDEX_OL=1) callers pass the GPU
    // flat array's max_o so the shadow and the flat array share encoding
    // and the rebuild upload is a single H2D copy. When the cuco-OL
    // path is active, callers pass 3000 (the initial-pop count per
    // district; equal to the shadow's largest o_id at load time).
    TpccCpuShadowIndex(TpccConfig config, uint32_t maxO_ol);
    ~TpccCpuShadowIndex();

    TpccCpuShadowIndex(const TpccCpuShadowIndex&) = delete;
    TpccCpuShadowIndex& operator=(const TpccCpuShadowIndex&) = delete;

    // Fill the five static shadows (W/D/C/I/S) and the initial population
    // of the three growing shadows (NO/O/OL). Idempotent: re-running
    // rebuilds the same content. Called from TpccDb::loadInitialData
    // before TpccGpuIndex::loadInitialData; the GPU bulk_inserts and the
    // host shadow fills are otherwise independent.
    void loadInitialData();

    // Per-epoch mirror update. Called by TpccGpuIndex::indexTxns after
    // the GPU cuco bulk_inserts complete and the D2H of insert keys
    // into h_no_keys() / h_o_keys() / h_ol_keys() finishes. The j-th
    // valid insert of each table is assigned CRID = initial_count +
    // old_free_start + j, which matches the GPU's CRID allocation (the
    // free-row arrays are populated in lex order by thrust::sequence).
    void mirrorEpoch(uint32_t num_new_orders_inserts,
                     uint32_t num_orders_inserts,
                     uint32_t num_order_lines_inserts,
                     uint32_t no_old_free_start,
                     uint32_t o_old_free_start,
                     uint32_t ol_old_free_start);

    // Snapshot shift at the start of every indexTxns. Captures the
    // free_start values that were durable at the end of epochs E-1 and
    // E-2, the safe rollback target throughout the entirety of epoch E.
    void shiftSnapshotsAtEpochStart(uint32_t new_order_free_start,
                                    uint32_t order_free_start,
                                    uint32_t order_line_free_start);

    // Erase straggler entries (CRID >= per-table max) from the three
    // growing shadows. Called by the rebuild path so the shadow matches
    // a fresh end-of-(E-2) state. Static shadows are immutable after
    // loadInitialData and need no pass.
    void eraseStragglers(uint32_t no_max, uint32_t o_max, uint32_t ol_max);

    // After a rollback rebuild to free_starts F, both prev1 and prev2
    // are set to F (no further rollback valid until two more epochs
    // have run).
    void syncSnapshotsToRollback(TpccFreeStarts target);

    // Recover-mode: rebuild the 3 growing-table insert shadows
    // from the durable (mmap'd) insert-key arrays for the cumulative insert
    // counts in `f` (= f_{E-2}). The j-th durable key of each table maps to
    // CRID base+j, matching mirrorEpoch's allocation. Used in a fresh process
    // before rebuildIndexesFromShadow. No-op when not in durable mode.
    void reconstructInsertsFromDurable(TpccFreeStarts f);

    // Per-epoch NewOrder delete mirror, the dual of mirrorEpoch's insert
    // append. Appends this epoch's delivered NO keys (already D2H'd into
    // h_no_delete_keys()) to the durable delete log at
    // [old_delete_count, old_delete_count+n).
    void mirrorEpochNoDeletes(uint32_t num_deletes, uint32_t old_delete_count);

    // Recover-mode: apply the first `count` durable NO delete-log entries
    // to the NewOrder shadow (sentinel each key's dense slot). Runs after
    // reconstructInsertsFromDurable; deletes are terminal (a delivered
    // (w,d,o) never recurs), so the two passes need no interleaving.
    void applyNoDeletesFromDurable(uint32_t count);

#ifdef EGAD_VALIDATION
    // Order-independent digest of the live NewOrder key->CRID mapping
    // derived from the durable logs alone (initial undelivered population
    // + first ins_count insert entries minus first del_count delete
    // entries). The delete-aware liveness gate: store byte hashes cannot
    // see a wrongly revived NO row because a delete writes no record
    // bytes. 0 when not in durable mode.
    uint64_t noLiveDigestFromLogs(uint32_t ins_count, uint32_t del_count) const;

    // Recover-time self-consistency check: the same digest computed over
    // the reconstructed NO shadow must equal an independent straight-line
    // replay of the logs at the rollback cursors. Logs [LIVE-CHECK] PASS
    // or FAILED; does not throw.
    void verifyNoLiveAgainstLogs(uint32_t ins_count, uint32_t del_count) const;
#endif // EGAD_VALIDATION

    // Read-only access to the underlying vectors. Used by the GPU index
    // rebuild upload path.
    const std::vector<uint32_t>& shadow_w()  const { return shadow_w_;  }
    const std::vector<uint32_t>& shadow_d()  const { return shadow_d_;  }
    const std::vector<uint32_t>& shadow_c()  const { return shadow_c_;  }
    const std::vector<uint32_t>& shadow_i()  const { return shadow_i_;  }
    const std::vector<uint32_t>& shadow_s()  const { return shadow_s_;  }
    const std::vector<uint32_t>& shadow_no() const { return shadow_no_; }
    const std::vector<uint32_t>& shadow_o()  const { return shadow_o_;  }
    const std::vector<uint32_t>& shadow_ol() const { return shadow_ol_; }

    // Dense-stride accessors.
    uint32_t max_o_orders() const { return max_o_orders_; }  // NO/O stride
    uint32_t max_o_ol()     const { return maxO_ol_;      }  // OL stride

    // Pinned host buffers consumed by the per-epoch mirror. Owned by
    // the shadow because they are sized to the shadow's needs and read
    // only by mirrorEpoch. TpccGpuIndex::indexTxns issues the D2H into
    // these buffers and then calls mirrorEpoch.
    NewOrderKey::baseType*  h_no_keys() { return h_no_keys_; }
    OrderKey::baseType*     h_o_keys()  { return h_o_keys_;  }
    OrderLineKey::baseType* h_ol_keys() { return h_ol_keys_; }
    NewOrderKey::baseType*  h_no_delete_keys() { return h_no_delete_keys_; }

private:
    TpccConfig tpcc_config_;
    uint32_t   maxO_ol_;            // OL dense stride (3000 or flat max_o)
    uint32_t   max_o_orders_ = 0;   // NO/O dense stride (>= 3000)

    std::vector<uint32_t> shadow_w_;
    std::vector<uint32_t> shadow_d_;
    std::vector<uint32_t> shadow_c_;
    std::vector<uint32_t> shadow_i_;
    std::vector<uint32_t> shadow_s_;
    std::vector<uint32_t> shadow_no_;
    std::vector<uint32_t> shadow_o_;
    std::vector<uint32_t> shadow_ol_;

    // Trailing snapshots for two-epoch rollback. Updated at the start of
    // every indexTxns via shiftSnapshotsAtEpochStart; throughout epoch E,
    // *_prev2_ holds f_{E-2}.
    uint32_t no_free_start_prev1_ = 0, no_free_start_prev2_ = 0;
    uint32_t o_free_start_prev1_  = 0, o_free_start_prev2_  = 0;
    uint32_t ol_free_start_prev1_ = 0, ol_free_start_prev2_ = 0;

    // Pinned host buffers for per-epoch D2H of insert keys (NO/O/OL).
    NewOrderKey::baseType*  h_no_keys_ = nullptr;
    OrderKey::baseType*     h_o_keys_  = nullptr;
    OrderLineKey::baseType* h_ol_keys_ = nullptr;

    // Durable (mmap-backed when EPIC_DURABLE_STORE/RECOVER_FROM)
    // append-only arrays of each growing table's insert keys, indexed by the
    // table's cumulative free_start (CRID - initial_pop). mirrorEpoch appends
    // each epoch; reconstructInsertsFromDurable rebuilds the dense shadows from
    // them in a fresh process. null in the normal path (no allocation/overhead).
    NewOrderKey::baseType*  durable_no_keys_ = nullptr;
    OrderKey::baseType*     durable_o_keys_  = nullptr;
    OrderLineKey::baseType* durable_ol_keys_ = nullptr;

    // Pinned host buffer for per-epoch D2H of delivered (deleted) NO keys.
    // Allocated only for Delivery-bearing mixes.
    NewOrderKey::baseType*  h_no_delete_keys_ = nullptr;

    // Durable append-only array of deleted NO keys, indexed by cumulative
    // delete count; the insert log's dual. Each NO row is delivered at
    // most once, so the log is bounded by the NO key universe. null when
    // not in durable mode or the mix has no Delivery.
    NewOrderKey::baseType*  durable_no_delete_keys_ = nullptr;
};

} // namespace epic::tpcc

#endif // EPIC_BENCHMARKS_TPCC_CPU_SHADOW_INDEX_H
