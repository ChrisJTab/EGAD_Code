//
// Created by Christian on 2026-05-23 (YCSB analog of TpccCpuShadowIndex).
//
// CPU-side durable shadow of the YCSB key->CRID mapping. Owned by
// YcsbBenchmark, lifetime independent of YcsbGpuIndex. Survives a GPU
// crash because it is plain host memory; serves as the rebuild source
// for the GPU cuco map during recovery.
//
// Mirrors the GPU primary index on inserts: every CRID admitted to the
// GPU also lands here under the same key. Sharded by (key %
// kShards) into kShards ankerl::unordered_dense maps so the per-epoch
// OMP fill has no cross-shard contention.
//
// Used by YcsbGpuIndex for two things:
//   1. smoke check after loadInitialData (sanity-check the shards landed
//      in the right place)
//   2. rebuildCucoFromShadow (recovery: repopulate the GPU cuco map
//      after a crash)

#ifndef EPIC_BENCHMARKS_YCSB_CPU_SHADOW_INDEX_H
#define EPIC_BENCHMARKS_YCSB_CPU_SHADOW_INDEX_H

#include <array>
#include <cstdint>

#include <ankerl/unordered_dense.h>

#include <benchmarks/ycsb_config.h>

namespace epic::ycsb {

class YcsbCpuShadowIndex
{
public:
    // Number of host-side shards. Picked so each OMP thread can own one
    // shard during the per-epoch mirror, with no cross-shard contention.
    static constexpr uint32_t kShards = 12;

    using Shard = ankerl::unordered_dense::map<uint32_t, uint32_t>;

    explicit YcsbCpuShadowIndex(YcsbConfig config);
    ~YcsbCpuShadowIndex();

    YcsbCpuShadowIndex(const YcsbCpuShadowIndex&) = delete;
    YcsbCpuShadowIndex& operator=(const YcsbCpuShadowIndex&) = delete;

    // Populate the shards with the initial record set (CRID == key for
    // keys in [0, starting_num_records)). Idempotent. Called from
    // YcsbBenchmark::loadInitialData before YcsbGpuIndex::loadInitialData
    // so the GPU index's optional post-load smoke check sees the shadow
    // populated.
    void loadInitialData();

    // Per-epoch mirror update. Called by YcsbGpuIndex::indexTxns after
    // the GPU cuco bulk_insert completes and the D2H of insert keys
    // into h_insert_keys() finishes. The j-th valid insert is assigned
    // CRID = starting_num_records + old_free_start + j, matching the
    // GPU's free-row allocation (dp_free_rows is thrust::sequence'd
    // starting at starting_num_records at load time).
    void mirrorEpoch(uint32_t num_inserts, uint32_t old_free_start);

    // Snapshot shift at the start of every indexTxns. Captures the
    // free_start values that were durable at the end of epochs E-1 and
    // E-2 so free_start_prev2_ holds the rollback target f_{E-2}
    // throughout the entirety of epoch E.
    void shiftSnapshotsAtEpochStart(uint32_t free_start);

    // Erase straggler entries (CRID >= max_crid) from every shard.
    // Called by the rebuild path so the shadow matches a fresh
    // end-of-(E-2) state.
    void eraseStragglers(uint32_t max_crid);

    // After a rollback rebuild to free_start F, both prev1 and prev2
    // are set to F (no further rollback valid until two more epochs
    // have run).
    void syncSnapshotsToRollback(uint32_t target);

    // Recover-mode: rebuild the insert shards from the durable
    // insert-keys array (mmap'd, survives the crash) for CRIDs [starting,
    // starting+count). The j-th durable key maps to CRID starting+j, matching
    // mirrorEpoch's allocation. Used in a fresh process before rebuildCucoFromShadow.
    void reconstructInsertsFromDurable(uint32_t count);

    // Read access for the GPU index's smoke check + rebuild upload path.
    const std::array<Shard, kShards>& shards() const { return shards_; }

    // Pinned host buffer consumed by the per-epoch mirror. Owned here
    // because it is sized to the shadow's needs and read only by
    // mirrorEpoch. YcsbGpuIndex::indexTxns issues the D2H into this
    // buffer and then calls mirrorEpoch.
    uint32_t* h_insert_keys() { return h_insert_keys_; }

    // Static accessors so external callers (the GPU index) can use the
    // same shard count and a matching shard() lookup helper.
    static uint32_t shardForKey(uint32_t key) { return key % kShards; }

private:
    YcsbConfig ycsb_config_;
    std::array<Shard, kShards> shards_;

    // Trailing snapshots for two-epoch rollback. Throughout epoch E,
    // free_start_prev2_ holds f_{E-2}.
    uint32_t free_start_prev1_ = 0;
    uint32_t free_start_prev2_ = 0;

    // Pinned host buffer for per-epoch D2H of insert keys.
    uint32_t* h_insert_keys_ = nullptr;

    // Durable (mmap-backed when EPIC_DURABLE_STORE/RECOVER_FROM)
    // append-only array of insert keys, indexed by (CRID - starting_num_records).
    // mirrorEpoch appends each epoch; recover rebuilds the shards from it. null
    // when not in durable mode (normal/throughput path -> no allocation/overhead).
    uint32_t* durable_insert_keys_ = nullptr;
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_CPU_SHADOW_INDEX_H
