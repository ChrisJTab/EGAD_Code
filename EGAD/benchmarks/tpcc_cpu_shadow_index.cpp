//
// Created by Christian on 2026-05-23.
//
// Implementation of TpccCpuShadowIndex. See the header for the rationale
// and the operation contracts.
//
// All operations are pure host code; the only CUDA touch points are the
// three pinned-host allocations for the per-epoch D2H landing buffers,
// routed through util_memory's Malloc/Free wrappers so this file stays
// .cpp (no nvcc dependency).
//

#include <benchmarks/tpcc_cpu_shadow_index.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>   // std::getenv (durable-mode gating)
#include <cstring>   // std::memcpy (durable insert-key append)
#include <vector>

#include <omp.h>

#include <benchmarks/recovery_verify.h>     // FNV primitives for the liveness gate (validation build)
#include <benchmarks/tpcc_flat_index.h>     // denseIdxOL
#include <benchmarks/tpcc_table.h>          // key types
#include <util_log.h>
#include <util_math.h>                      // formatSizeBytes
#include <util_memory.h>                    // Malloc / Free (pinned host)

namespace epic::tpcc {

namespace {

// Per-district max o_id stride for NO/O. Must be >= the largest runtime
// o_id; derived from the larger of orderTableSize and newOrderTableSize
// so the same scalar works for both. Round up. Guaranteed >= 3000 (the
// initial-pop count per district) since both table sizes are at least
// W * 10 * 3000.
uint32_t computeOrdersMaxO(const TpccConfig& cfg)
{
    const uint64_t denom = static_cast<uint64_t>(cfg.num_warehouses) * 10ull;
    const uint64_t o_total  = static_cast<uint64_t>(cfg.orderTableSize());
    const uint64_t no_total = static_cast<uint64_t>(cfg.newOrderTableSize());
    const uint64_t total = std::max(o_total, no_total);
    return static_cast<uint32_t>((total + denom - 1ull) / denom);
}

} // namespace

TpccCpuShadowIndex::TpccCpuShadowIndex(TpccConfig config, uint32_t maxO_ol)
    : tpcc_config_(config)
    , maxO_ol_(maxO_ol)
{
    auto& logger = Logger::GetInstance();

    max_o_orders_ = computeOrdersMaxO(tpcc_config_);

    const uint64_t W = static_cast<uint64_t>(tpcc_config_.num_warehouses);
    const uint64_t maxO_no = static_cast<uint64_t>(max_o_orders_);
    const uint64_t maxO_ol_u = static_cast<uint64_t>(maxO_ol_);

    shadow_w_ .assign(static_cast<size_t>(W),                              kSentinel);
    shadow_d_ .assign(static_cast<size_t>(W * 10ull),                      kSentinel);
    shadow_c_ .assign(static_cast<size_t>(W * 10ull * 3000ull),            kSentinel);
    shadow_i_ .assign(static_cast<size_t>(100000),                         kSentinel);
    shadow_s_ .assign(static_cast<size_t>(W * 100000ull),                  kSentinel);
    shadow_no_.assign(static_cast<size_t>(W * 10ull * maxO_no),            kSentinel);
    shadow_o_ .assign(static_cast<size_t>(W * 10ull * maxO_no),            kSentinel);
    shadow_ol_.assign(static_cast<size_t>(W * 10ull * maxO_ol_u * 15ull),  kSentinel);

    const size_t no_bytes = static_cast<size_t>(tpcc_config_.num_txns)      * sizeof(NewOrderKey::baseType);
    const size_t o_bytes  = static_cast<size_t>(tpcc_config_.num_txns)      * sizeof(OrderKey::baseType);
    const size_t ol_bytes = static_cast<size_t>(tpcc_config_.num_txns) * 15 * sizeof(OrderLineKey::baseType);
    h_no_keys_ = static_cast<NewOrderKey::baseType*> (Malloc(no_bytes));
    h_o_keys_  = static_cast<OrderKey::baseType*>    (Malloc(o_bytes));
    h_ol_keys_ = static_cast<OrderLineKey::baseType*>(Malloc(ol_bytes));

    // In durable mode, back each growing table's insert-key
    // history with a durable (mmap'd) array so a fresh process can rebuild the
    // insert shadows after a crash. Sized to each table's insert headroom
    // (tableSize - initial_pop); indexed by cumulative free_start. null in the
    // normal/throughput path -> no allocation, no overhead. Mirrors YCSB's
    // durable_insert_keys_ pattern, one array per growing table.
    const bool durable = (std::getenv("EPIC_DURABLE_STORE") != nullptr ||
                          std::getenv("EPIC_RECOVER_FROM") != nullptr);
    if (durable) {
        const uint64_t no_init = W * 10ull * 900ull;
        const uint64_t o_init  = W * 10ull * 3000ull;
        const uint64_t ol_init = W * 10ull * 3000ull * 15ull;
        const size_t no_head = static_cast<size_t>(static_cast<uint64_t>(tpcc_config_.newOrderTableSize())  - no_init);
        const size_t o_head  = static_cast<size_t>(static_cast<uint64_t>(tpcc_config_.orderTableSize())     - o_init);
        const size_t ol_head = static_cast<size_t>(static_cast<uint64_t>(tpcc_config_.orderLineTableSize()) - ol_init);
        durable_no_keys_ = static_cast<NewOrderKey::baseType*> (MallocDurable("tpcc_shadow_no_keys", no_head * sizeof(NewOrderKey::baseType)));
        durable_o_keys_  = static_cast<OrderKey::baseType*>    (MallocDurable("tpcc_shadow_o_keys",  o_head  * sizeof(OrderKey::baseType)));
        durable_ol_keys_ = static_cast<OrderLineKey::baseType*>(MallocDurable("tpcc_shadow_ol_keys", ol_head * sizeof(OrderLineKey::baseType)));
        logger.Info("durable shadow insert-key arrays (headroom entries): NO={} O={} OL={}",
                    no_head, o_head, ol_head);
    }

    // NewOrder delete-log allocations, only for Delivery-bearing
    // hybrid_staging runs (the delete path is hybrid-only; the baseline
    // modes stay pure Epic). Each NO row is delivered (deleted) at most
    // once, so the durable log's bound is the NO key universe.
    if (tpcc_config_.execution_mode == ExecMode::HYBRID_STAGING
        && tpcc_config_.txn_mix.delivery > 0) {
        h_no_delete_keys_ = static_cast<NewOrderKey::baseType*>(
            Malloc(static_cast<size_t>(tpcc_config_.num_txns) * 10 * sizeof(NewOrderKey::baseType)));
        if (durable) {
            durable_no_delete_keys_ = static_cast<NewOrderKey::baseType*>(
                MallocDurable("tpcc_shadow_no_del_keys",
                              static_cast<size_t>(tpcc_config_.newOrderTableSize()) *
                                  sizeof(NewOrderKey::baseType)));
        }
    }

    const size_t total_mb =
        (shadow_w_.size() + shadow_d_.size() + shadow_c_.size() + shadow_i_.size() + shadow_s_.size()
         + shadow_no_.size() + shadow_o_.size() + shadow_ol_.size()) * sizeof(uint32_t) / (1024ull * 1024ull);
    logger.Info("[SHADOW] CPU shadow indexes allocated: {} MB total "
                "(W={} D={} C={} I={} S={} NO={} O={} OL={} entries, max_o_ol={} max_o_orders={})",
        total_mb,
        shadow_w_.size(), shadow_d_.size(), shadow_c_.size(), shadow_i_.size(), shadow_s_.size(),
        shadow_no_.size(), shadow_o_.size(), shadow_ol_.size(), maxO_ol_, max_o_orders_);
}

TpccCpuShadowIndex::~TpccCpuShadowIndex()
{
    if (h_no_keys_) { Free(h_no_keys_); h_no_keys_ = nullptr; }
    if (h_o_keys_)  { Free(h_o_keys_);  h_o_keys_  = nullptr; }
    if (h_ol_keys_) { Free(h_ol_keys_); h_ol_keys_ = nullptr; }
    if (h_no_delete_keys_) { Free(h_no_delete_keys_); h_no_delete_keys_ = nullptr; }
}

void TpccCpuShadowIndex::loadInitialData()
{
    const uint32_t W       = tpcc_config_.num_warehouses;
    const uint32_t maxO_no = max_o_orders_;
    const uint32_t maxO    = maxO_ol_;

    // W: one CRID per warehouse, dense_idx == CRID.
    #pragma omp parallel for
    for (uint32_t w_id = 1; w_id <= W; ++w_id) {
        shadow_w_[denseIdxW(w_id)] = w_id - 1u;
    }

    // D: dense_idx == CRID by construction.
    #pragma omp parallel for collapse(2)
    for (uint32_t w_id = 1; w_id <= W; ++w_id) {
        for (uint32_t d_id = 1; d_id <= 10; ++d_id) {
            shadow_d_[denseIdxD(w_id, d_id)] = (w_id - 1u) * 10u + (d_id - 1u);
        }
    }

    // C: dense_idx == CRID by construction.
    #pragma omp parallel for collapse(2)
    for (uint32_t w_id = 1; w_id <= W; ++w_id) {
        for (uint32_t d_id = 1; d_id <= 10; ++d_id) {
            const uint32_t base = ((w_id - 1u) * 10u + (d_id - 1u)) * 3000u;
            for (uint32_t c_id = 1; c_id <= 3000; ++c_id) {
                shadow_c_[denseIdxC(w_id, d_id, c_id)] = base + (c_id - 1u);
            }
        }
    }

    // I: dense_idx == CRID.
    #pragma omp parallel for
    for (uint32_t i_id = 1; i_id <= 100000; ++i_id) {
        shadow_i_[denseIdxI(i_id)] = i_id - 1u;
    }

    // S: dense_idx == CRID by construction.
    #pragma omp parallel for collapse(2)
    for (uint32_t w_id = 1; w_id <= W; ++w_id) {
        for (uint32_t i_id = 1; i_id <= 100000; ++i_id) {
            shadow_s_[denseIdxS(w_id, i_id)] = (w_id - 1u) * 100000u + (i_id - 1u);
        }
    }

    // NO/O/OL initial populations.
    //   Order:      all 3000 per (w,d) load; CRID == dense_idx.
    //   NewOrder:   only o_id in (2100, 3000] load (900 per (w,d)).
    //               CRID = (w-1)*9000 + (d-1)*900 + (o-2101). The shared
    //               NO key space is sized to W*10*max_o so the unloaded
    //               slots (o<=2100) stay at sentinel.
    //   OrderLine:  15 per o for all 3000 o per (w,d). CRID and dense_idx
    //               match when max_o == 3000; the EPIC_FLAT_INDEX_OL path
    //               may use max_o > 3000 to leave runtime headroom, which
    //               just leaves gaps in the shadow array, no correctness
    //               issue.
    #pragma omp parallel for collapse(2)
    for (uint32_t w_id = 1; w_id <= W; ++w_id) {
        for (uint32_t d_id = 1; d_id <= 10; ++d_id) {
            const uint32_t o_base  = ((w_id - 1u) * 10u + (d_id - 1u)) * 3000u;
            const uint32_t no_base = ((w_id - 1u) * 10u + (d_id - 1u)) * 900u;
            const uint32_t ol_base = ((w_id - 1u) * 10u + (d_id - 1u)) * 3000u * 15u;
            for (uint32_t o_id = 1; o_id <= 3000; ++o_id) {
                shadow_o_[denseIdxO(w_id, d_id, o_id, maxO_no)] = o_base + (o_id - 1u);
                if (o_id > 2100) {
                    shadow_no_[denseIdxNO(w_id, d_id, o_id, maxO_no)] = no_base + (o_id - 2101u);
                }
                for (uint32_t ol_n = 1; ol_n <= 15; ++ol_n) {
                    shadow_ol_[denseIdxOL(w_id, d_id, o_id, ol_n, maxO)] =
                        ol_base + (o_id - 1u) * 15u + (ol_n - 1u);
                }
            }
        }
    }
}

void TpccCpuShadowIndex::mirrorEpoch(uint32_t num_new_orders_inserts,
                                     uint32_t num_orders_inserts,
                                     uint32_t num_order_lines_inserts,
                                     uint32_t no_old_free_start,
                                     uint32_t o_old_free_start,
                                     uint32_t ol_old_free_start)
{
    // Append this epoch's insert keys to the durable arrays at
    // [old_free_start, old_free_start+n) so durable[j] -> CRID (base+j) survives
    // a crash. The recover process (EPIC_RECOVER_FROM) reconstructs the in-memory
    // dense shadow from these via reconstructInsertsFromDurable; the live path
    // only appends here.
    // CRID for each per-table insert is initial_count + old_free_start + j,
    // matching the GPU's free-row allocation. Cheap (O(inserts/epoch)); durable only.
    if (durable_no_keys_) {
        std::memcpy(durable_no_keys_ + no_old_free_start, h_no_keys_,
                    static_cast<size_t>(num_new_orders_inserts) * sizeof(NewOrderKey::baseType));
        std::memcpy(durable_o_keys_ + o_old_free_start, h_o_keys_,
                    static_cast<size_t>(num_orders_inserts) * sizeof(OrderKey::baseType));
        std::memcpy(durable_ol_keys_ + ol_old_free_start, h_ol_keys_,
                    static_cast<size_t>(num_order_lines_inserts) * sizeof(OrderLineKey::baseType));
    }
}

// See header. Mirror of mirrorEpoch's scatter, run over the
// cumulative insert counts in `f` with old_free_start = 0 (the j-th durable key
// of each table -> CRID base+j). OMP-parallel OL loop (dominant), sequential
// NO/O. Used by recover-mode before rebuildIndexesFromShadow. No-op when the
// durable arrays were not allocated (normal path).
void TpccCpuShadowIndex::reconstructInsertsFromDurable(TpccFreeStarts f)
{
    if (!durable_no_keys_) return;
    const uint32_t W       = tpcc_config_.num_warehouses;
    const uint32_t no_base = W * 10u * 900u;
    const uint32_t o_base  = W * 10u * 3000u;
    const uint32_t ol_base = W * 10u * 3000u * 15u;
    const uint32_t maxO    = maxO_ol_;
    const uint32_t maxO_no = max_o_orders_;

    #pragma omp parallel for num_threads(8)
    for (uint32_t j = 0; j < f.order_line; ++j) {
        OrderLineKey k; k.base_key = durable_ol_keys_[j];
        shadow_ol_[denseIdxOL(k.ol_w_id, k.ol_d_id, k.ol_o_id, k.ol_number, maxO)] = ol_base + j;
    }
    for (uint32_t j = 0; j < f.new_order; ++j) {
        NewOrderKey k; k.base_key = durable_no_keys_[j];
        shadow_no_[denseIdxNO(k.no_w_id, k.no_d_id, k.no_o_id, maxO_no)] = no_base + j;
    }
    for (uint32_t j = 0; j < f.order; ++j) {
        OrderKey k; k.base_key = durable_o_keys_[j];
        shadow_o_[denseIdxO(k.o_w_id, k.o_d_id, k.o_id, maxO_no)] = o_base + j;
    }
    Logger::GetInstance().Info("[RECOVER] reconstructed shadow inserts from durable arrays: NO={} O={} OL={}",
                               f.new_order, f.order, f.order_line);
}

void TpccCpuShadowIndex::mirrorEpochNoDeletes(uint32_t num_deletes, uint32_t old_delete_count)
{
    if (num_deletes == 0) return;

    // Append this epoch's delivered NO keys to the durable delete log at
    // [old_delete_count, old_delete_count+num_deletes). The recover
    // process re-applies them via applyNoDeletesFromDurable up to the
    // marker's delete cursor; the live path only appends here.
    if (durable_no_delete_keys_) {
        std::memcpy(durable_no_delete_keys_ + old_delete_count, h_no_delete_keys_,
                    static_cast<size_t>(num_deletes) * sizeof(NewOrderKey::baseType));
    }
}

// See header. Sentinel each deleted key's dense slot; the dual of
// reconstructInsertsFromDurable's NO pass.
void TpccCpuShadowIndex::applyNoDeletesFromDurable(uint32_t count)
{
    if (!durable_no_delete_keys_ || count == 0) return;

    uint32_t first = 0;
#ifdef EGAD_VALIDATION
    // Negative control for the liveness gate: drop the first delete-log
    // entry so its NO row is wrongly revived. verifyNoLiveAgainstLogs
    // must catch the divergence; the store byte hashes cannot (a delete
    // writes no record bytes). No-op unless the env hook is set.
    if (std::getenv("EPIC_RECOVERY_DROP_ONE_DELETE") != nullptr) {
        Logger::GetInstance().Info(
            "[NEG-CONTROL] dropping NO delete-log entry 0 on recover");
        first = 1;
    }
#endif // EGAD_VALIDATION

    const uint32_t maxO_no = max_o_orders_;
    for (uint32_t j = first; j < count; ++j) {
        NewOrderKey k; k.base_key = durable_no_delete_keys_[j];
        shadow_no_[denseIdxNO(k.no_w_id, k.no_d_id, k.no_o_id, maxO_no)] = kSentinel;
    }
    Logger::GetInstance().Info("[RECOVER] applied {} NO delete entries from durable shadow log", count);
}

#ifdef EGAD_VALIDATION

namespace {
// Per-entry fold for the NO liveness digest: FNV over (dense_idx, crid).
// Combined across entries by sum, so the digest is order-independent and
// comparable between a shadow walk and an independent log replay.
inline uint64_t foldNoLiveEntry(uint32_t dense_idx, uint32_t crid)
{
    uint64_t h = fnv1aUpdate(kFnvOffsetBasis, &dense_idx, sizeof(dense_idx));
    return fnv1aUpdate(h, &crid, sizeof(crid));
}
} // namespace

// Straight-line replay of the NO logs into a scratch state: initial
// undelivered population, then inserts [0, ins_count), then deletes
// [0, del_count). Deliberately independent of loadInitialData /
// reconstructInsertsFromDurable / applyNoDeletesFromDurable so it can
// catch bugs in any of them.
uint64_t TpccCpuShadowIndex::noLiveDigestFromLogs(uint32_t ins_count, uint32_t del_count) const
{
    if ((ins_count > 0 && !durable_no_keys_) || (del_count > 0 && !durable_no_delete_keys_)) return 0;

    const uint32_t W = tpcc_config_.num_warehouses;
    const uint32_t maxO_no = max_o_orders_;
    std::vector<uint32_t> state(shadow_no_.size(), kSentinel);
    for (uint32_t w = 1; w <= W; ++w) {
        for (uint32_t d = 1; d <= 10; ++d) {
            const uint32_t no_base = ((w - 1u) * 10u + (d - 1u)) * 900u;
            for (uint32_t o = 2101; o <= 3000; ++o) {
                state[denseIdxNO(w, d, o, maxO_no)] = no_base + (o - 2101u);
            }
        }
    }
    const uint32_t no_init = W * 10u * 900u;
    for (uint32_t j = 0; j < ins_count; ++j) {
        NewOrderKey k; k.base_key = durable_no_keys_[j];
        state[denseIdxNO(k.no_w_id, k.no_d_id, k.no_o_id, maxO_no)] = no_init + j;
    }
    for (uint32_t j = 0; j < del_count; ++j) {
        NewOrderKey k; k.base_key = durable_no_delete_keys_[j];
        state[denseIdxNO(k.no_w_id, k.no_d_id, k.no_o_id, maxO_no)] = kSentinel;
    }

    uint64_t acc = 0;
    #pragma omp parallel for reduction(+:acc) schedule(static)
    for (size_t i = 0; i < state.size(); ++i) {
        if (state[i] != kSentinel) acc += foldNoLiveEntry(static_cast<uint32_t>(i), state[i]);
    }
    return acc;
}

void TpccCpuShadowIndex::verifyNoLiveAgainstLogs(uint32_t ins_count, uint32_t del_count) const
{
    auto& logger = Logger::GetInstance();
    uint64_t shadow_acc = 0;
    size_t shadow_count = 0;
    #pragma omp parallel for reduction(+:shadow_acc) reduction(+:shadow_count) schedule(static)
    for (size_t i = 0; i < shadow_no_.size(); ++i) {
        if (shadow_no_[i] != kSentinel) {
            shadow_acc += foldNoLiveEntry(static_cast<uint32_t>(i), shadow_no_[i]);
            ++shadow_count;
        }
    }
    const uint64_t log_acc = noLiveDigestFromLogs(ins_count, del_count);
    const size_t expected_count =
        static_cast<size_t>(tpcc_config_.num_warehouses) * 10u * 900u + ins_count - del_count;
    if (shadow_acc == log_acc && shadow_count == expected_count) {
        logger.Info("[LIVE-CHECK] PASS NO shadow-vs-log at ins={} del={}: live={} digest=0x{:016x}",
                    ins_count, del_count, shadow_count, shadow_acc);
    } else {
        logger.Error("[LIVE-CHECK] FAILED NO shadow-vs-log at ins={} del={}: "
                     "shadow live={} digest=0x{:016x} vs log live={} digest=0x{:016x}",
                     ins_count, del_count, shadow_count, shadow_acc, expected_count, log_acc);
    }
}

#endif // EGAD_VALIDATION

void TpccCpuShadowIndex::shiftSnapshotsAtEpochStart(uint32_t new_order_free_start,
                                                   uint32_t order_free_start,
                                                   uint32_t order_line_free_start)
{
    no_free_start_prev2_ = no_free_start_prev1_;
    no_free_start_prev1_ = new_order_free_start;
    o_free_start_prev2_  = o_free_start_prev1_;
    o_free_start_prev1_  = order_free_start;
    ol_free_start_prev2_ = ol_free_start_prev1_;
    ol_free_start_prev1_ = order_line_free_start;
}

void TpccCpuShadowIndex::eraseStragglers(uint32_t no_max, uint32_t o_max, uint32_t ol_max)
{
    auto& logger = Logger::GetInstance();
    size_t erased_no = 0, erased_o = 0, erased_ol = 0;
    #pragma omp parallel sections num_threads(3)
    {
        #pragma omp section
        { size_t c = 0; for (auto& v : shadow_no_) if (v != kSentinel && v >= no_max) { v = kSentinel; ++c; } erased_no = c; }
        #pragma omp section
        { size_t c = 0; for (auto& v : shadow_o_ ) if (v != kSentinel && v >= o_max ) { v = kSentinel; ++c; } erased_o  = c; }
        #pragma omp section
        { size_t c = 0; for (auto& v : shadow_ol_) if (v != kSentinel && v >= ol_max) { v = kSentinel; ++c; } erased_ol = c; }
    }
    logger.Info("[SHADOW-REBUILD] erased stragglers: NO={} O={} OL={}",
                erased_no, erased_o, erased_ol);
}

void TpccCpuShadowIndex::syncSnapshotsToRollback(TpccFreeStarts target)
{
    no_free_start_prev1_ = no_free_start_prev2_ = target.new_order;
    o_free_start_prev1_  = o_free_start_prev2_  = target.order;
    ol_free_start_prev1_ = ol_free_start_prev2_ = target.order_line;
}

} // namespace epic::tpcc
