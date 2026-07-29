//
// tpcc_recovery.cpp -- process-restart recovery for TpccDb, split out of
// tpcc.cpp for separation/readability. These are TpccDb member definitions
// (full access to private members; no friend needed). Contains the Primary
// Store version rollback
// (rollbackPrimaryStoreVersions), the durable crash marker (bumpRecoveryMarker),
// and the recovery/determinism verification (growingInsertCounts,
// hashCpuRecords, logCpuRecordsTableHashes, verifyInsertedRecords). tpcc.cpp
// keeps construction + the normal per-epoch pipeline.
//
#include "benchmarks/tpcc.h"

#include <cassert>
#include <random>
#include <cstdio>
#include <cstring>
#include <thread>
#include <sstream>
#include <chrono>

#include "benchmarks/tpcc_txn.h"

#include "util_log.h"
#include "util_device_type.h"
#include "util_exec_mode.h"
#include "gpu_txn.h"
#include "gpu_allocator.h"
#include "gpu_execution_planner.h"
#include "benchmarks/tpcc_gpu_submitter.h"
#include "benchmarks/tpcc_executor.h"
#include "benchmarks/tpcc_gpu_executor.h"
#include <benchmarks/tpcc_txn_gen.h>
#include <benchmarks/tpcc_gpu_index.h>
#include <benchmarks/tpcc_cpu_executor.h>
#include <benchmarks/tpcc_flat_index.h>  // flatOLEnabled, computeOLMaxO
#include <benchmarks/recovery_verify.h>  // shared byte-exact verify primitives

// Hybrid staging: the per-table stager template (instances live in TpccDb).
#include "benchmarks/tpcc_hybrid_stager.h"
#include "benchmarks/tpcc_hybrid_remap.h"
#include "benchmarks/tpcc_hybrid_executor.h"
#include "benchmarks/tpcc_staging_capacities.h"
#include "benchmarks/tpcc_workload_aware_capacities.h"

namespace epic::tpcc {

// Demote the Primary Store to end-of-(E-2) before replaying E-1.
// After a crash (or controlled reset), each record's two slots hold its two
// most-recent durable versions, which include epoch E-1's value (tag E-1).
// Replaying E-1 with that state present makes the executor's write-slot picker
// (write the OLDER slot, gpu_storage.cuh) overwrite the end-of-(E-2) value,
// collapsing both slots to the E-1 value and losing the E-2 version for any
// record not rewritten in epoch E or later. Zeroing every tag >= E-1 demotes
// the E-1/E slot to "oldest" so the surviving <= E-2 slot is read as the base
// and the demoted slot becomes the write target; replay then re-derives E-1
// (and E) deterministically and overwrites it. The marker discipline (the
// bump runs after E-1's writeback fully drains and before E's is submitted)
// guarantees at most one slot per record carries a tag >= E-1 at any crash
// point, so the <= E-2 value always survives. Value bytes are left in place (replay
// overwrites the demoted slot). Recovery path only; the hot executor is
// untouched. growing_extent = the growing-table cursors as of end-of-(E-1).
void TpccDb::rollbackPrimaryStoreVersions(uint32_t min_discard_epoch,
                                          const TpccFreeStarts& growing_extent)
{
    auto& logger = Logger::GetInstance();
    auto rollback_table = [&](auto* recs, uint64_t count) -> uint64_t {
        if (!recs || count == 0) return 0;
        uint64_t demoted = 0;
#pragma omp parallel for reduction(+:demoted) schedule(static)
        for (uint64_t i = 0; i < count; ++i) {
            if (recs[i].version1 >= min_discard_epoch) { recs[i].version1 = 0; ++demoted; }
            if (recs[i].version2 >= min_discard_epoch) { recs[i].version2 = 0; ++demoted; }
        }
        return demoted;
    };
    const uint64_t no_initial = static_cast<uint64_t>(config.num_warehouses) * 10ull * 900ull;
    const uint64_t o_initial  = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull;
    const uint64_t ol_initial = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull * 15ull;
    uint64_t d = 0;
    d += rollback_table(cpu_records.warehouse_record,  config.warehouseTableSize());
    d += rollback_table(cpu_records.district_record,   config.districtTableSize());
    d += rollback_table(cpu_records.customer_record,   config.customerTableSize());
    d += rollback_table(cpu_records.item_record,       config.itemTableSize());
    d += rollback_table(cpu_records.stock_record,      config.stockTableSize());
    d += rollback_table(cpu_records.new_order_record,  no_initial + growing_extent.new_order);
    d += rollback_table(cpu_records.order_record,      o_initial  + growing_extent.order);
    d += rollback_table(cpu_records.order_line_record, ol_initial + growing_extent.order_line);
    logger.Info("Primary Store version rollback: demoted {} slots with tag >= {}",
                d, min_discard_epoch);
}

// Recovery-stable growing-table insert counts. The per-stager
// cache_slots_insert_allocate_total_ counter starts at 0 in the fresh
// stagers a recover process constructs, so after recovery it only
// reflects post-recovery epochs and under-sizes the growing tables for
// hashing. The index's free-list cursors (getInsertCounts) ARE restored
// across recovery via rebuildIndexesFromShadow(saved_counts), and equal
// the cumulative runtime insert count per growing table
// (CRID = initial_pop + free_start + j). Prefer
// the index cursors; fall back to the stager counter if the index is not the
// GPU variant (e.g. CPU-only paths) or not yet built.
TpccDb::TpccGrowingCounts TpccDb::growingInsertCounts() const
{
    TpccGrowingCounts gc{};
    auto *gpu_index_ptr =
        dynamic_cast<TpccGpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>*>(index.get());
    if (gpu_index_ptr) {
        const TpccFreeStarts fs = gpu_index_ptr->getInsertCounts();
        gc.new_order  = fs.new_order;
        gc.order      = fs.order;
        gc.order_line = fs.order_line;
        return gc;
    }
    gc.new_order  = new_order_stager  ? new_order_stager->cache_slots_insert_allocate_total()  : 0;
    gc.order      = order_stager      ? order_stager->cache_slots_insert_allocate_total()      : 0;
    gc.order_line = order_line_stager ? order_line_stager->cache_slots_insert_allocate_total() : 0;
    return gc;
}

// Advance the durable crash marker + shift the growing-table
// cursor history. Called once per epoch at the end of runEpoch(E), after
// every stager's sync_flush has retired E-1 and before E's writeback is
// submitted -- the one point where end-of-(E-1) is fully durable and no
// end-of-(E-2) slot has been overwritten, so both the old marker (E) and the
// new one (E+1) describe a recoverable store. Phases 0-2 of epoch E see the
// value set by runEpoch(E-1) (= E); phases 3 and 5 see E+1. The shift leaves
// p2 == f_{crash_epoch-2} (the rollback target); the index cursors it
// reads advance only during indexTxns, long before any bump position in the
// epoch. No-op on non-durable runs. The banked prepare/publish split keeps
// the marker coherent if the process dies mid-update (recovery_meta.h) and
// gives the torn-marker probe its injection point.
void TpccDb::bumpRecoveryMarker(uint32_t crash_epoch)
{
    if (!recovery_meta_) return;
    const TpccGrowingCounts gc = growingInsertCounts();
    auto *gpu_index_ptr =
        dynamic_cast<TpccGpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>*>(index.get());
    const uint32_t no_del = gpu_index_ptr ? gpu_index_ptr->getNoDeleteCount() : 0u;
    const uint64_t word = recovery_meta_->prepare(crash_epoch,
        [&](TpccRecoveryCursors& nb, const TpccRecoveryCursors* ob) {
            nb.no_p2 = ob ? ob->no_p1 : 0u; nb.no_p1 = static_cast<uint32_t>(gc.new_order);
            nb.o_p2  = ob ? ob->o_p1  : 0u; nb.o_p1  = static_cast<uint32_t>(gc.order);
            nb.ol_p2 = ob ? ob->ol_p1 : 0u; nb.ol_p1 = static_cast<uint32_t>(gc.order_line);
            nb.no_d_p2 = ob ? ob->no_d_p1 : 0u; nb.no_d_p1 = no_del;
        });
#ifdef EGAD_VALIDATION
    maybeTearMarkerAt(crash_epoch);
#endif
    recovery_meta_->publishPrepared(word);
}

#ifdef EGAD_VALIDATION  // recovery state-hash + negative-control hooks (validation only)

uint64_t TpccDb::hashCpuRecords()
{
    constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
    constexpr uint64_t kFnvPrime       = 0x100000001b3ULL;

    // Parallel FNV-1a over one table buffer's first `count` records.
    auto hash_table = [&](const void *base, size_t rec_size, uint64_t count) -> uint64_t {
        if (!base || count == 0) return 0;
        const uint8_t *data = reinterpret_cast<const uint8_t *>(base);
        uint64_t acc = 0;
#pragma omp parallel for reduction(^:acc) schedule(static)
        for (uint64_t r = 0; r < count; ++r) {
            uint64_t h = kFnvOffsetBasis;
            const uint8_t *rec = data + r * rec_size;
            for (size_t b = 0; b < rec_size; ++b) { h ^= rec[b]; h *= kFnvPrime; }
            acc ^= h;
        }
        return acc;
    };

    // Growing-table in-use counts mirror verifyInsertedRecords: initial
    // population + cumulative insert-allocate admissions.
    const uint64_t no_initial = static_cast<uint64_t>(config.num_warehouses) * 10ull * 900ull;
    const uint64_t o_initial  = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull;
    const uint64_t ol_initial = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull * 15ull;
    const TpccGrowingCounts gc = growingInsertCounts();
    const uint64_t no_n = no_initial + gc.new_order;
    const uint64_t o_n  = o_initial  + gc.order;
    const uint64_t ol_n = ol_initial + gc.order_line;

    // Fold per-table hashes in a fixed order (xor-then-multiply so two tables
    // with identical bytes do not cancel each other out).
    uint64_t h = kFnvOffsetBasis;
    auto fold = [&](uint64_t th) { h ^= th; h *= kFnvPrime; };
    fold(hash_table(cpu_records.warehouse_record,  sizeof(Record<WarehouseValue>),  config.warehouseTableSize()));
    fold(hash_table(cpu_records.district_record,   sizeof(Record<DistrictValue>),   config.districtTableSize()));
    fold(hash_table(cpu_records.customer_record,   sizeof(Record<CustomerValue>),   config.customerTableSize()));
    fold(hash_table(cpu_records.item_record,       sizeof(Record<ItemValue>),       config.itemTableSize()));
    fold(hash_table(cpu_records.stock_record,      sizeof(Record<StockValue>),      config.stockTableSize()));
    fold(hash_table(cpu_records.new_order_record,  sizeof(Record<NewOrderValue>),   no_n));
    fold(hash_table(cpu_records.order_record,      sizeof(Record<OrderValue>),      o_n));
    fold(hash_table(cpu_records.order_line_record, sizeof(Record<OrderLineValue>),  ol_n));
    return h;
}

// Per-table positional + multiset hashes (see header).
void TpccDb::logCpuRecordsTableHashes()
{
    constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
    constexpr uint64_t kFnvPrime       = 0x100000001b3ULL;
    auto &logger = Logger::GetInstance();

    // Per table (typed Record<ValueType>*), three hashes over the first `count`
    // records:
    //   positional : index-sensitive xor-fold (detects layout/placement change)
    //   full_ms    : multiset (sum) of full-record hashes (values + version tags)
    //   val_ms     : multiset (sum) of VALUES-ONLY hashes, with the two value
    //                slots sorted -> invariant to version tags AND slot swaps.
    // full_ms differs but val_ms matches => tag/slot/placement only (value-equivalent).
    // val_ms differs                     => genuinely different values.
    //
    // combined_valms folds the per-table val_ms in fixed table order into one
    // placement-invariant digest ([STATE-HASH-VALMS]). It is the recovery GO/NO-GO
    // gate: it ignores internal CRID placement / dual-version slot pick (which are
    // legitimately non-deterministic for inserts under eviction) and compares only
    // the logical database content, which is what deterministic replay must match.
    uint64_t combined_valms = kFnvOffsetBasis;
    auto log_table = [&](const char *name, auto *recs, uint64_t count) {
        if (!recs || count == 0) {
            logger.Info("[TBL-HASH] {} count=0", name);
            combined_valms ^= 0ull; combined_valms *= kFnvPrime;
            return;
        }
        const size_t vsz = sizeof(recs->value1);   // ValueType size
        const size_t rsz = sizeof(*recs);          // full Record size
        uint64_t positional = 0, full_ms = 0, val_ms = 0;
        uint64_t nonzero = 0; // records with any non-zero value byte (written-ness probe)
#pragma omp parallel for reduction(^:positional) reduction(+:full_ms) reduction(+:val_ms) reduction(+:nonzero) schedule(static)
        for (uint64_t r = 0; r < count; ++r) {
            const auto &rec = recs[r];
            const uint8_t *rb = reinterpret_cast<const uint8_t *>(&rec);
            uint64_t hf = kFnvOffsetBasis;
            for (size_t b = 0; b < rsz; ++b) { hf ^= rb[b]; hf *= kFnvPrime; }
            // values-only, slot-sorted (lower value bytes first) -> tag/slot
            // invariant. Shared, byte-exact fold (recovery_verify.h).
            const uint8_t *v1 = reinterpret_cast<const uint8_t *>(&rec.value1);
            const uint8_t *v2 = reinterpret_cast<const uint8_t *>(&rec.value2);
            const uint64_t hv = foldRecordValuesOnly(rec);
            bool nz = false;
            for (size_t b = 0; b < vsz; ++b) { if (v1[b] || v2[b]) { nz = true; break; } }
            positional ^= (hf ^ (r * 0x9E3779B97F4A7C15ULL));
            full_ms    += hf;
            val_ms     += hv;
            nonzero    += nz ? 1 : 0;
        }
        logger.Info("[TBL-HASH] {} count={} nonzero={} positional=0x{:016x} full_ms=0x{:016x} val_ms=0x{:016x}",
                    name, count, nonzero, positional, full_ms, val_ms);
        combined_valms ^= val_ms; combined_valms *= kFnvPrime;
    };

    const uint64_t no_initial = static_cast<uint64_t>(config.num_warehouses) * 10ull * 900ull;
    const uint64_t o_initial  = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull;
    const uint64_t ol_initial = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull * 15ull;
    const TpccGrowingCounts gc = growingInsertCounts();
    const uint64_t no_n = no_initial + gc.new_order;
    const uint64_t o_n  = o_initial  + gc.order;
    const uint64_t ol_n = ol_initial + gc.order_line;

    log_table("warehouse",  cpu_records.warehouse_record,  config.warehouseTableSize());
    log_table("district",   cpu_records.district_record,   config.districtTableSize());
    log_table("customer",   cpu_records.customer_record,   config.customerTableSize());
    log_table("item",       cpu_records.item_record,       config.itemTableSize());
    log_table("stock",      cpu_records.stock_record,      config.stockTableSize());
    log_table("new_order",  cpu_records.new_order_record,  no_n);
    log_table("order",      cpu_records.order_record,      o_n);
    log_table("order_line", cpu_records.order_line_record, ol_n);

    // Placement-invariant recovery gate: identical across two same-seed runs iff
    // the logical database content matches (internal placement/slot ignored).
    logger.Info("[STATE-HASH-VALMS] tpcc cpu_records placement-invariant = 0x{:016x}", combined_valms);
}

// VERIFICATION-ONLY negative-control hook. Strictly a NO-OP unless one
// of the two env vars below is set, so OFF leaves the Primary Store byte-identical
// (the recovery determinism gate keeps holding). When set, it perturbs the final
// CPU Primary Store right BEFORE the end-of-run state hashes are computed, to prove
// the hash gate actually CATCHES a corrupted recovery (a smoke-detector test
// button). Targets the stock table, which is always fully populated (no inserts,
// no eviction holes) so a fixed low CRID is guaranteed live.
//   EPIC_RECOVERY_CORRUPT_ONE : XOR 0xFF into ONE value byte of stock CRID 0.
//   EPIC_RECOVERY_SWAP_TWO    : swap the FULL value bytes (value1 AND value2)
//                               between stock CRID 0 and CRID 1.
void TpccDb::corruptPrimaryStoreForNegControl()
{
    // Thin wrapper over the shared, byte-exact negative-control primitive
    // (recovery_verify.h). Targets the stock table, which is always fully
    // populated (no inserts, no eviction holes) so a fixed low CRID is live.
    negControlPerturb(cpu_records.stock_record, config.stockTableSize(), "stock");
}

#endif // EGAD_VALIDATION

void TpccDb::verifyInsertedRecords()
{
    auto& logger = Logger::GetInstance();
    if (config.execution_mode != ExecMode::HYBRID_STAGING) {
        logger.Info("[VERIFY] skipping: not hybrid_staging");
        return;
    }
    if (!new_order_stager || !order_stager || !order_line_stager) {
        logger.Info("[VERIFY] skipping: stagers not constructed");
        return;
    }

    const uint32_t max_epoch = static_cast<uint32_t>(config.epochs);

    // Per-table verification: scan recs[insert_first .. insert_first +
    // insert_count) and check each slot has a non-zero version. insert_first
    // is the table's initial-population size (the GPU indexer mints fresh
    // CRIDs starting there — see tpcc_gpu_index.cu's free_rows init around
    // line 916). insert_count is the cumulative number of insert-allocate
    // admissions across all epochs (the stager accumulates this in
    // cache_slots_insert_allocate_total_).
    auto verify_one = [&](const char* name, auto* recs,
                          uint64_t insert_first, uint64_t insert_count) -> bool {
        if (insert_count == 0) {
            logger.Info("[VERIFY] {}: no insert CRIDs to verify", name);
            return true;
        }
        uint64_t no_version = 0;
        uint64_t epoch_oob  = 0;
        uint64_t v1_only = 0, v2_only = 0, both = 0;
        for (uint64_t i = 0; i < insert_count; ++i) {
            const uint64_t crid = insert_first + i;
            const uint32_t v1 = recs[crid].version1;
            const uint32_t v2 = recs[crid].version2;
            if (v1 == 0 && v2 == 0) {
                ++no_version;
                if (no_version <= 5) {
                    logger.Error("[VERIFY] {} insert CRID {} has v1=0 v2=0 — executor INSERT did not reach CPU primary store",
                                 name, crid);
                }
                continue;
            }
            if (v1 > max_epoch || v2 > max_epoch) {
                ++epoch_oob;
                if (epoch_oob <= 5) {
                    logger.Error("[VERIFY] {} insert CRID {} has out-of-range version (v1={} v2={} max_epoch={})",
                                 name, crid, v1, v2, max_epoch);
                }
                continue;
            }
            if (v1 != 0 && v2 != 0) ++both;
            else if (v1 != 0)       ++v1_only;
            else                    ++v2_only;
        }
        const uint64_t failures = no_version + epoch_oob;
        if (failures == 0) {
            logger.Info("[VERIFY] {} OK: all {} insert CRIDs have a valid version on CPU primary store "
                        "(v1_only={} v2_only={} both_versions={})",
                        name, insert_count, v1_only, v2_only, both);
            return true;
        }
        logger.Error("[VERIFY] {} FAILED: {}/{} insert CRIDs broken (missing_version={} epoch_oob={})",
                     name, failures, insert_count, no_version, epoch_oob);
        return false;
    };

    // Per-table initial population sizes — match tpcc_gpu_index.cu's
    // loadInitialData. The first insert mints CRID = initial_pop.
    const uint64_t no_initial = static_cast<uint64_t>(config.num_warehouses) * 10ull * 900ull;
    const uint64_t o_initial  = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull;
    const uint64_t ol_initial = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull * 15ull;

    bool all_ok = true;
    all_ok &= verify_one("NewOrder",  cpu_records.new_order_record,  no_initial,
                         new_order_stager->cache_slots_insert_allocate_total());
    all_ok &= verify_one("Order",     cpu_records.order_record,      o_initial,
                         order_stager->cache_slots_insert_allocate_total());
    all_ok &= verify_one("OrderLine", cpu_records.order_line_record, ol_initial,
                         order_line_stager->cache_slots_insert_allocate_total());
    if (all_ok) {
        logger.Info("[VERIFY] all 3 growing-table insert verifiers passed");
    } else {
        logger.Error("[VERIFY] one or more growing-table insert verifiers FAILED");
    }
}

} // namespace epic::tpcc
