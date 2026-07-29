//
// Created by Christian on 2026-05-23 (YCSB analog of TpccCpuShadowIndex). See the header for the rationale.
//
// All operations are pure host code; the only CUDA touch point is the
// pinned-host allocation for the per-epoch D2H landing buffer, routed
// through util_memory's Malloc/Free wrappers so this file stays .cpp
// (no nvcc dependency).
//

#include <benchmarks/ycsb_cpu_shadow_index.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <omp.h>

#include <benchmarks/recovery_verify.h>  // FNV primitives for the liveness gate (validation build)
#include <util_log.h>
#include <util_memory.h>  // Malloc / Free (pinned host)

namespace epic::ycsb {

YcsbCpuShadowIndex::YcsbCpuShadowIndex(YcsbConfig config)
    : ycsb_config_(config)
{
    // Per-shard reservation: ~num_records / kShards entries per shard
    // (load is uniform because the shard key is key % kShards). Reserves
    // up front so the per-epoch mirror inserts do not trigger a rehash.
    for (auto& s : shards_) {
        s.reserve(ycsb_config_.num_records / kShards + 1);
    }

    h_insert_keys_ = static_cast<uint32_t*>(
        Malloc(static_cast<size_t>(ycsb_config_.num_txns) *
               ycsb_config_.num_ops_per_txn * sizeof(uint32_t)));

    // In durable mode, back the insert-keys history with a
    // durable (mmap'd) array so a fresh process can rebuild the insert shards
    // after a crash. Indexed by (CRID - starting_num_records). null otherwise
    // (normal/throughput path -> no allocation, no overhead).
    const bool durable = std::getenv("EPIC_DURABLE_STORE") || std::getenv("EPIC_RECOVER_FROM");
    if (durable && ycsb_config_.num_records > ycsb_config_.starting_num_records) {
        const size_t headroom = ycsb_config_.num_records - ycsb_config_.starting_num_records;
        durable_insert_keys_ = static_cast<uint32_t*>(
            MallocDurable("ycsb_shadow_keys", headroom * sizeof(uint32_t)));
    }

    // Delete-log allocations, only for delete-bearing mixes. A key is
    // deleted at most once (deletes are terminal), so the durable log's
    // bound is the total key universe.
    if (ycsb_config_.txn_mix.num_deletes > 0) {
        h_delete_keys_ = static_cast<uint32_t*>(
            Malloc(static_cast<size_t>(ycsb_config_.num_txns) *
                   ycsb_config_.num_ops_per_txn * sizeof(uint32_t)));
        if (durable) {
            durable_delete_keys_ = static_cast<uint32_t*>(
                MallocDurable("ycsb_shadow_del_keys",
                              static_cast<size_t>(ycsb_config_.num_records) * sizeof(uint32_t)));
        }
    }
}

YcsbCpuShadowIndex::~YcsbCpuShadowIndex()
{
    if (h_insert_keys_) {
        Free(h_insert_keys_);
        h_insert_keys_ = nullptr;
    }
    if (h_delete_keys_) {
        Free(h_delete_keys_);
        h_delete_keys_ = nullptr;
    }
}

void YcsbCpuShadowIndex::loadInitialData()
{
    auto& logger = Logger::GetInstance();
    auto t0 = std::chrono::high_resolution_clock::now();

    // OMP per-shard fill: thread tid owns shard tid, walks keys
    // [tid, tid + kShards, ...) up to starting_num_records and inserts
    // (key, key) since CRID == key for the initial population.
    #pragma omp parallel num_threads(kShards)
    {
        int tid = omp_get_thread_num();
        auto& shard = shards_[tid];
        for (uint32_t k = static_cast<uint32_t>(tid);
             k < ycsb_config_.starting_num_records;
             k += kShards)
        {
            shard.try_emplace(k, k);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    logger.Info("[SHADOW] CPU shadow populated in {} ms (sharded, {} threads, {} entries)",
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
                kShards, ycsb_config_.starting_num_records);

    // Smoke check: a handful of keys must round-trip through the right
    // shard. Catches accidental shard-encoding mismatches between
    // load and lookup paths.
    for (uint32_t sample : {0u, 1u, 100u, 10000u}) {
        if (sample >= ycsb_config_.starting_num_records) continue;
        const auto& shard = shards_[shardForKey(sample)];
        auto it = shard.find(sample);
        if (it == shard.end() || it->second != sample) {
            throw std::runtime_error("CPU shadow smoke check failed at key " +
                                     std::to_string(sample));
        }
    }
}

void YcsbCpuShadowIndex::mirrorEpoch(uint32_t num_inserts, uint32_t old_free_start)
{
    if (num_inserts == 0) return;

    // Append this epoch's insert keys to the durable array at
    // [old_free_start, old_free_start+num_inserts) so durable[j] -> CRID
    // (starting+j) survives a crash. The recover process reconstructs the
    // in-memory shard maps from these via reconstructInsertsFromDurable; the
    // live path only appends here.
    if (durable_insert_keys_) {
        std::memcpy(durable_insert_keys_ + old_free_start, h_insert_keys_,
                    static_cast<size_t>(num_inserts) * sizeof(uint32_t));
    }
}

// See header. Mirror of mirrorEpoch's CRID allocation: the
// j-th durable insert key maps to CRID starting+j. OMP per-shard (no cross-
// shard contention). Used by recover-mode before rebuildCucoFromShadow.
void YcsbCpuShadowIndex::reconstructInsertsFromDurable(uint32_t count)
{
    if (!durable_insert_keys_ || count == 0) return;
    const uint32_t starting = ycsb_config_.starting_num_records;
    #pragma omp parallel num_threads(kShards)
    {
        int tid = omp_get_thread_num();
        auto& shard = shards_[tid];
        for (uint32_t j = 0; j < count; ++j) {
            uint32_t key = durable_insert_keys_[j];
            if (shardForKey(key) == static_cast<uint32_t>(tid)) {
                shard.try_emplace(key, starting + j);
            }
        }
    }
    Logger::GetInstance().Info("[RECOVER] reconstructed {} insert entries from durable shadow array", count);
}

void YcsbCpuShadowIndex::mirrorEpochDeletes(uint32_t num_deletes, uint32_t old_delete_count)
{
    if (num_deletes == 0) return;

    // Append this epoch's delete keys to the durable log at
    // [old_delete_count, old_delete_count+num_deletes). The recover process
    // re-applies them via applyDeletesFromDurable up to the marker's delete
    // cursor; the live path only appends here. The in-memory shards are not
    // touched: they are reconstructed from the logs on recover, and nothing
    // reads them during a normal run.
    if (durable_delete_keys_) {
        std::memcpy(durable_delete_keys_ + old_delete_count, h_delete_keys_,
                    static_cast<size_t>(num_deletes) * sizeof(uint32_t));
    }
}

// See header. OMP per-shard like reconstructInsertsFromDurable: each
// thread erases the log's keys that hash to its shard.
void YcsbCpuShadowIndex::applyDeletesFromDurable(uint32_t count)
{
    if (!durable_delete_keys_ || count == 0) return;

    uint32_t first = 0;
#ifdef EGAD_VALIDATION
    // Negative control for the liveness gate: drop the first delete-log
    // entry so its key is wrongly revived. verifyLiveAgainstLogs must
    // catch the divergence; the store byte hashes cannot (a delete writes
    // no record bytes). No-op unless the env hook is set.
    if (std::getenv("EPIC_RECOVERY_DROP_ONE_DELETE") != nullptr) {
        Logger::GetInstance().Info(
            "[NEG-CONTROL] dropping delete-log entry 0 (key {}) on recover", durable_delete_keys_[0]);
        first = 1;
    }
#endif // EGAD_VALIDATION

    #pragma omp parallel num_threads(kShards)
    {
        int tid = omp_get_thread_num();
        auto& shard = shards_[tid];
        for (uint32_t j = first; j < count; ++j) {
            uint32_t key = durable_delete_keys_[j];
            if (shardForKey(key) == static_cast<uint32_t>(tid)) {
                shard.erase(key);
            }
        }
    }
    Logger::GetInstance().Info("[RECOVER] applied {} delete entries from durable shadow log", count);
}

#ifdef EGAD_VALIDATION

namespace {
// Per-entry fold for the liveness digest: FNV over the (key, crid) pair.
// Combined across entries by sum, so the digest is order-independent and
// comparable between a shard walk and a log replay.
inline uint64_t foldLiveEntry(uint32_t key, uint32_t crid)
{
    uint64_t h = fnv1aUpdate(kFnvOffsetBasis, &key, sizeof(key));
    return fnv1aUpdate(h, &crid, sizeof(crid));
}
} // namespace

uint64_t YcsbCpuShadowIndex::liveDigestFromLogs(uint32_t ins_count, uint32_t del_count) const
{
    if (del_count > 0 && !durable_delete_keys_) return 0;
    if (ins_count > 0 && !durable_insert_keys_) return 0;

    ankerl::unordered_dense::set<uint32_t> deleted;
    deleted.reserve(del_count);
    for (uint32_t j = 0; j < del_count; ++j) {
        deleted.insert(durable_delete_keys_[j]);
    }

    const uint32_t starting = static_cast<uint32_t>(ycsb_config_.starting_num_records);
    uint64_t acc = 0;
    #pragma omp parallel for reduction(+:acc) schedule(static)
    for (uint32_t k = 0; k < starting; ++k) {
        if (!deleted.contains(k)) acc += foldLiveEntry(k, k);
    }
    for (uint32_t j = 0; j < ins_count; ++j) {
        const uint32_t key = durable_insert_keys_[j];
        if (!deleted.contains(key)) acc += foldLiveEntry(key, starting + j);
    }
    return acc;
}

void YcsbCpuShadowIndex::verifyLiveAgainstLogs(uint32_t ins_count, uint32_t del_count) const
{
    auto& logger = Logger::GetInstance();
    uint64_t shard_acc = 0;
    size_t shard_count = 0;
    for (const auto& shard : shards_) {
        shard_count += shard.size();
        for (const auto& kv : shard) {
            shard_acc += foldLiveEntry(kv.first, kv.second);
        }
    }
    const uint64_t log_acc = liveDigestFromLogs(ins_count, del_count);
    const size_t expected_count =
        ycsb_config_.starting_num_records + ins_count - del_count;
    if (shard_acc == log_acc && shard_count == expected_count) {
        logger.Info("[LIVE-CHECK] PASS shadow-vs-log at ins={} del={}: live={} digest=0x{:016x}",
                    ins_count, del_count, shard_count, shard_acc);
    } else {
        logger.Error("[LIVE-CHECK] FAILED shadow-vs-log at ins={} del={}: "
                     "shadow live={} digest=0x{:016x} vs log live={} digest=0x{:016x}",
                     ins_count, del_count, shard_count, shard_acc, expected_count, log_acc);
    }
}

#endif // EGAD_VALIDATION

void YcsbCpuShadowIndex::shiftSnapshotsAtEpochStart(uint32_t free_start)
{
    free_start_prev2_ = free_start_prev1_;
    free_start_prev1_ = free_start;
}

void YcsbCpuShadowIndex::eraseStragglers(uint32_t max_crid)
{
    auto& logger = Logger::GetInstance();
    size_t erased = 0;
    for (auto& shard : shards_) {
        auto it = shard.begin();
        while (it != shard.end()) {
            if (it->second >= max_crid) {
                it = shard.erase(it);
                ++erased;
            } else {
                ++it;
            }
        }
    }
    logger.Info("[SHADOW-REBUILD] erased stragglers: {} entries", erased);
}

void YcsbCpuShadowIndex::syncSnapshotsToRollback(uint32_t target)
{
    free_start_prev1_ = target;
    free_start_prev2_ = target;
}

} // namespace epic::ycsb
