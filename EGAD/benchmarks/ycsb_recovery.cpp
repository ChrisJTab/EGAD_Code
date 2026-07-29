//
// ycsb_recovery.cpp -- process-restart recovery for YcsbBenchmark, split out
// of ycsb.cpp for separation/readability. These are YcsbBenchmark member
// definitions (full access to private members; no friend needed). Contains
// the Primary Store version rollback
// (rollbackPrimaryStoreVersions), the durable crash marker (bumpRecoveryMarker),
// and the recovery/determinism verification (hashCpuRecords,
// hashCpuRecordsValOnly, verifyInsertedRecords). ycsb.cpp keeps construction +
// the normal per-epoch pipeline.
//
#include <algorithm>
#include <cstdlib>   // std::getenv (neg-control hook)
#include <cstring>   // std::memcmp (placement-invariant values-only hash)
#include <random>
#include <util_log.h>
#include <benchmarks/ycsb.h>
#include <benchmarks/ycsb_txn.h>
#include <util_random_zipfian.h>
#include <benchmarks/ycsb_index.h>
#include <benchmarks/ycsb_gpu_index.h>
#include <gpu_allocator.h>
#include <gpu_execution_planner.h>
#include <benchmarks/ycsb_gpu_submitter.h>
#include <benchmarks/ycsb_gpu_executor.h>
#include <benchmarks/ycsb_hybrid_executor.h>
#include <benchmarks/ycsb_cpu_executor.h>
#include <util_gpu_transfer.h>
#include <util_memory.h>
#include <benchmarks/recovery_verify.h>  // shared byte-exact verify primitives

namespace epic::ycsb {

#ifdef EGAD_VALIDATION  // recovery state-hash + negative-control hooks (validation only)

// FNV-1a 64-bit hash over CPU_records. Pure host code; reads
// the variant's typed pointer, then walks the byte range. Used to verify
// byte-identical state between no-crash and crash+replay runs.
uint64_t YcsbBenchmark::hashCpuRecords() const
{
    constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
    constexpr uint64_t kFnvPrime       = 0x100000001b3ULL;

    const size_t rec_size  = config.split_field ? sizeof(ycsb::YcsbFieldRecords)
                                                : sizeof(ycsb::YcsbRecords);
    const size_t num_units = static_cast<size_t>(config.num_records) *
                             (config.split_field ? 10u : 1u);

    uint64_t result = 0;
    std::visit([&](auto* p) {
        if (!p) return;
        const uint8_t* data = reinterpret_cast<const uint8_t*>(p);
        uint64_t acc = 0;
#pragma omp parallel for reduction(^:acc) schedule(static)
        for (size_t r = 0; r < num_units; ++r) {
            uint64_t h = kFnvOffsetBasis;
            const uint8_t* rec = data + r * rec_size;
            for (size_t b = 0; b < rec_size; ++b) {
                h ^= rec[b];
                h *= kFnvPrime;
            }
            acc ^= h;
        }
        result = acc;
    }, CPU_records);

    return result;
}

uint64_t YcsbBenchmark::hashCpuRecordsValOnly() const
{
    // FNV-1a constants live in recovery_verify.h; the per-record value-only fold
    // (foldRecordValuesOnly) carries them. This function only owns the orchestration
    // (which records, the multiset combine).
    const size_t num_units = static_cast<size_t>(config.num_records) *
                             (config.split_field ? 10u : 1u);

    uint64_t result = 0;
    std::visit([&](auto* p) {
        if (!p) return;
        uint64_t acc = 0;
#pragma omp parallel for reduction(+:acc) schedule(static)
        for (size_t r = 0; r < num_units; ++r) {
            // values-only, slot-sorted (lower value bytes first) -> invariant
            // to version tags AND value1<->value2 slot swaps. Shared, byte-exact
            // fold (recovery_verify.h); the same fold backs TPCC's val_ms.
            acc += foldRecordValuesOnly(p[r]);   // sum -> multiset, position-independent
        }
        result = acc;
    }, CPU_records);

    return result;
}

// VERIFICATION-ONLY negative-control hook. Strictly a NO-OP unless one
// of the two env vars is set, so OFF leaves the Primary Store byte-identical.
// Perturbs the final CPU Primary Store right BEFORE the end-of-run hashes to prove
// the hash gate CATCHES a corrupted recovery. Record 0 is always populated.
// Uses std::visit so it compiles for both YcsbRecords* and YcsbFieldRecords*.
//   EPIC_RECOVERY_CORRUPT_ONE : XOR 0xFF into ONE value byte of record 0.
//   EPIC_RECOVERY_SWAP_TWO    : swap full value bytes (value1 AND value2) of
//                               record 0 and record 1.
void YcsbBenchmark::corruptPrimaryStoreForNegControl()
{
    // Thin wrapper over the shared, byte-exact negative-control primitive
    // (recovery_verify.h). std::visit dispatches to the active variant so the
    // primitive instantiates for both YcsbRecords* and YcsbFieldRecords*.
    // Record 0 is always populated.
    const size_t num_units = static_cast<size_t>(config.num_records) *
                             (config.split_field ? 10u : 1u);
    std::visit([&](auto* p) { negControlPerturb(p, num_units, "ycsb"); }, CPU_records);
}

#endif // EGAD_VALIDATION

// Demote the Primary Store to end-of-(E-2) before replaying E-1.
// Identical rationale to TpccDb::rollbackPrimaryStoreVersions: with the store
// at end-of-(E-1), replaying E-1 makes the executor's write-slot picker (writes
// the OLDER slot) clobber the end-of-(E-2) value, since E-1's own output is
// already present. Zeroing every tag >= E-1 demotes that slot to "oldest" so the
// surviving <= E-2 slot is the read base and the demoted slot is the write
// target; replay then re-derives E-1 (and E) and overwrites it. YCSB-F is all
// updates, so this matters even though it has no inserts. Sweeps all CPU_records
// units (num_records x {10 if split_field else 1}); not-yet-inserted slots carry
// tag 0 and are no-ops. Value bytes are left in place (replay overwrites them).
void YcsbBenchmark::rollbackPrimaryStoreVersions(uint32_t min_discard_epoch)
{
    const size_t num_units = static_cast<size_t>(config.num_records) *
                             (config.split_field ? 10u : 1u);
    uint64_t demoted = 0;
    std::visit([&](auto* p) {
        if (!p) return;
        uint64_t d = 0;
#pragma omp parallel for reduction(+:d) schedule(static)
        for (size_t r = 0; r < num_units; ++r) {
            if (p[r].version1 >= min_discard_epoch) { p[r].version1 = 0; ++d; }
            if (p[r].version2 >= min_discard_epoch) { p[r].version2 = 0; ++d; }
        }
        demoted = d;
    }, CPU_records);
    Logger::GetInstance().Info("Primary Store version rollback: demoted {} slots with tag >= {}",
                               demoted, min_discard_epoch);
}

// Current insert cursor (0 if not the GPU index). YCSB-F has no
// inserts so this is always 0; YCSB-i advances it as records are inserted.
uint32_t YcsbBenchmark::currentInsertCount() const
{
    auto* gpu_index_ptr = dynamic_cast<YcsbGpuIndex*>(index.get());
    return gpu_index_ptr ? gpu_index_ptr->getInsertCount() : 0u;
}

// Advance the durable crash marker + shift the insert-cursor
// history. Called once per epoch at the end of runEpoch(E), after sync_flush
// has retired E-1 and before E's writeback is started -- the one point where
// end-of-(E-1) is fully durable and no end-of-(E-2) slot has been
// overwritten, so both the old marker (E) and the new one (E+1) describe a
// recoverable store. Phases 0-2 of epoch E see the value set by
// runEpoch(E-1) (= E); phases 3 and 5 see E+1. The shift leaves p2 ==
// f_{crash_epoch-2} (the rollback target); the insert cursor it reads
// advances only during indexTxns, long before any bump position in the
// epoch. No-op on non-durable runs. The banked prepare/publish split keeps
// the marker coherent if the process dies mid-update (recovery_meta.h) and
// gives the torn-marker probe its injection point.
void YcsbBenchmark::bumpRecoveryMarker(uint32_t crash_epoch)
{
    if (!recovery_meta_) return;
    const uint32_t insert_count = currentInsertCount();
    const uint64_t word = recovery_meta_->prepare(crash_epoch,
        [&](YcsbRecoveryCursors& nb, const YcsbRecoveryCursors* ob) {
            nb.free_start_p2 = ob ? ob->free_start_p1 : 0u;
            nb.free_start_p1 = insert_count;
        });
#ifdef EGAD_VALIDATION
    maybeTearMarkerAt(crash_epoch);
#endif
    recovery_meta_->publishPrepared(word);
}

void YcsbBenchmark::verifyInsertedRecords()
{
    auto& logger = Logger::GetInstance();

    if (config.execution_mode != ExecMode::HYBRID_STAGING) {
        return;
    }
    if (config.split_field) {
        // The split-field layout splits each logical record into 10 field
        // records (Record<YcsbFieldValue>) at offsets [crid*10, crid*10+10).
        // Verifying it requires matching the per-field admission/writeback
        // path; the prototype scope skips field-split.
        logger.Info("[VERIFY] skipping: split_field=true not supported by the inserted-record verifier");
        return;
    }
    auto* gpu_index = dynamic_cast<YcsbGpuIndex*>(index.get());
    if (!gpu_index) {
        logger.Info("[VERIFY] skipping: index is not a YcsbGpuIndex");
        return;
    }

    const uint32_t insert_count = gpu_index->getInsertCount();
    if (insert_count == 0) {
        logger.Info("[VERIFY] no insert CRIDs to verify");
        return;
    }

    auto* recs = std::get<YcsbRecords*>(CPU_records);
    const uint32_t insert_first = static_cast<uint32_t>(config.starting_num_records);
    const uint32_t max_epoch = static_cast<uint32_t>(config.epochs);

    uint32_t no_version = 0;
    uint32_t epoch_oob = 0;
    uint32_t v1_only = 0, v2_only = 0, both = 0;

    for (uint32_t i = 0; i < insert_count; ++i) {
        const uint32_t crid = insert_first + i;
        const uint32_t v1 = recs[crid].version1;
        const uint32_t v2 = recs[crid].version2;

        if (v1 == 0 && v2 == 0) {
            ++no_version;
            if (no_version <= 5) {
                logger.Error("[VERIFY] insert CRID {} has v1=0 v2=0 — executor INSERT did not reach CPU primary store", crid);
            }
            continue;
        }
        if (v1 > max_epoch || v2 > max_epoch) {
            ++epoch_oob;
            if (epoch_oob <= 5) {
                logger.Error("[VERIFY] insert CRID {} has out-of-range version (v1={} v2={} max_epoch={})",
                             crid, v1, v2, max_epoch);
            }
            continue;
        }
        if (v1 != 0 && v2 != 0) ++both;
        else if (v1 != 0)       ++v1_only;
        else                    ++v2_only;
    }

    const uint32_t failures = no_version + epoch_oob;
    if (failures == 0) {
        logger.Info("[VERIFY] OK: all {} insert CRIDs have a valid version on CPU primary store "
                    "(v1_only={} v2_only={} both_versions={})",
                    insert_count, v1_only, v2_only, both);
    } else {
        logger.Error("[VERIFY] FAILED: {}/{} insert CRIDs broken (missing_version={} epoch_oob={})",
                     failures, insert_count, no_version, epoch_oob);
    }
}

} // namespace epic::ycsb
