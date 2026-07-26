//
// Created by Shujian Qian on 2023-08-15.
//

#ifndef TPCC_H
#define TPCC_H

#include "tpcc_config.h"
#include "tpcc_common.h"
#include "tpcc_table.h"
#include "tpcc_txn.h"
#include <benchmarks/tpcc_storage.h>
#include "benchmarks/tpcc_submitter.h"

#include <vector>
#include <memory>

#include "execution_planner.h"
#include "table.h"
#include "txn_bridge.h"
#include <benchmarks/benchmark.h>
#include <benchmarks/tpcc_executor.h>
#include <benchmarks/tpcc_gpu_executor.h>
#include <benchmarks/tpcc_index.h>
#include <benchmarks/tpcc_cpu_aux_index.h>
#include <benchmarks/tpcc_cpu_shadow_index.h>
#include <benchmarks/tpcc_gpu_aux_index.h>
#include <gpu_allocator.h>
#include <benchmarks/tpcc_hybrid_stager.h>

namespace epic::tpcc {

// Convenience aliases — one TpccHybridStager type per writeable table.
// History gets no stager because the executor doesn't read or write it.
using WarehouseStager = TpccHybridStager<Record<WarehouseValue>, Version<WarehouseValue>>;
using DistrictStager  = TpccHybridStager<Record<DistrictValue>,  Version<DistrictValue>>;
using CustomerStager  = TpccHybridStager<Record<CustomerValue>,  Version<CustomerValue>>;
using ItemStager      = TpccHybridStager<Record<ItemValue>,      Version<ItemValue>>;
using StockStager     = TpccHybridStager<Record<StockValue>,     Version<StockValue>>;
using NewOrderStager  = TpccHybridStager<Record<NewOrderValue>,  Version<NewOrderValue>>;
using OrderStager     = TpccHybridStager<Record<OrderValue>,     Version<OrderValue>>;
using OrderLineStager = TpccHybridStager<Record<OrderLineValue>, Version<OrderLineValue>>;

class TpccDb : public Benchmark
{
private:
    TableExecutionPlanner *planner;

    Table warehouse_table;

    TpccConfig config;
    std::vector<TpccTxnArrayT> txn_array;
    TpccTxnArrayT index_input;
    TpccTxnParamArrayT index_output;
    TpccTxnParamArrayT initialization_input;
    TpccTxnExecPlanArrayT initialization_output;
    TpccTxnParamArrayT execution_param_input;
    TpccTxnExecPlanArrayT execution_plan_input;

    // Addaptors needed to get some of the inputs to work
    PackedTxnBridge input_index_bridge; // figures out if we need to move data around during the execution between the CPU and GPU
    PackedTxnBridge index_initialization_bridge;
    PackedTxnBridge index_execution_param_bridge;
    PackedTxnBridge initialization_execution_plan_bridge;

    // CPU-side durable shadow of the key->CRID mapping. Owned here, with
    // a lifetime independent of the GPU index, which holds a reference
    // to it. Only constructed for the GPU index path; null for the
    // CPU-index path (no shadow there).
    std::unique_ptr<TpccCpuShadowIndex> cpu_shadow_;

    std::shared_ptr<TpccIndex<TpccTxnArrayT, TpccTxnParamArrayT>> index;
    // index matches the key to the rowID + handles the allocation and deletion of new rows

    std::shared_ptr<TableExecutionPlanner> warehouse_planner;
    std::shared_ptr<TableExecutionPlanner> district_planner;
    std::shared_ptr<TableExecutionPlanner> customer_planner;
    std::shared_ptr<TableExecutionPlanner> history_planner;
    std::shared_ptr<TableExecutionPlanner> new_order_planner;
    std::shared_ptr<TableExecutionPlanner> order_planner;
    std::shared_ptr<TableExecutionPlanner> order_line_planner;
    std::shared_ptr<TableExecutionPlanner> item_planner;
    std::shared_ptr<TableExecutionPlanner> stock_planner;
    std::shared_ptr<TpccSubmitter<TpccTxnParamArrayT>> submitter;

    TpccRecords records; // GPU records (primary in gpu_only / cpu_only; cache pool in hybrid_staging)
    TpccVersions versions;

    // hybrid_staging-only: CPU primary store, addressed by CRID. Pre-allocated
    // pinned host memory via cudaHostAlloc(cudaHostAllocMapped); NUMA-bound to
    // node 0 (single-NUMA box). Sized to the CRID universe of each table.
    // History gets no allocation (no stager).
    TpccRecords cpu_records;

    // 8 per-table stagers (one per writeable table). History excluded.
    std::shared_ptr<WarehouseStager> warehouse_stager;
    std::shared_ptr<DistrictStager>  district_stager;
    std::shared_ptr<CustomerStager>  customer_stager;
    std::shared_ptr<ItemStager>      item_stager;
    std::shared_ptr<StockStager>     stock_stager;
    std::shared_ptr<NewOrderStager>  new_order_stager;
    std::shared_ptr<OrderStager>     order_stager;
    std::shared_ptr<OrderLineStager> order_line_stager;

    // 8 per-table FlushHandles for async flush (overlap_flush mode).
    // One handle per stager; never cross-wired (each stager's eviction
    // pins must come from its own handle's d_grids list, otherwise a
    // worker's in-flight slot can be selected as a rename victim by
    // its own next-epoch eviction). FlushHandle holds a std::thread,
    // mutex, and condition_variable, so it is non-copyable and non-
    // movable — these are named members rather than a vector.
    FlushHandle warehouse_flush;
    FlushHandle district_flush;
    FlushHandle customer_flush;
    FlushHandle item_flush;
    FlushHandle stock_flush;
    FlushHandle new_order_flush;
    FlushHandle order_flush;
    FlushHandle order_line_flush;

    // Allocator owned by TpccDb so the per-stager Allocator& references stay
    // valid for the lifetime of the stagers.
    std::unique_ptr<GpuAllocator> hybrid_allocator;

    std::shared_ptr<Executor<TpccTxnParamArrayT, TpccTxnExecPlanArrayT>> executor; // executes all the transactions

    TpccCpuAuxIndex cpu_aux_index; // used for tpcc range queries
    // GPU auxiliary B-link tree index. Held via unique_ptr so the
    // recovery rebuild can replace it with a fresh instance (the old
    // one's device buffers are freed explicitly first; see the
    // recover-mode block in runBenchmark).
    std::unique_ptr<TpccGpuAuxIndex<TpccTxnArrayT, TpccTxnParamArrayT>> gpu_aux_index;
    TpccPackedTxnArrayBuilder packed_txn_array_builder;

    // Durable crash-recovery metadata (mmap-backed when
    // EPIC_DURABLE_STORE/RECOVER_FROM is set; null on normal runs -> no overhead).
    // crash_epoch is advanced to "the next epoch to run" each time an epoch's
    // writeback is queued (bumpRecoveryMarker), so on a crash it names the epoch E
    // whose predecessor's writeback is in-flight; recovery rolls the Primary Store
    // back to end-of-(E-2) and replays E-1 and E. The 3 cursor pairs carry
    // f_{E-1}/f_{E-2} for the growing tables (prev2 == f_{E-2} = the rollback target).
    struct RecoveryMeta {
        uint32_t magic; uint32_t crash_epoch;
        uint32_t no_p1, no_p2, o_p1, o_p2, ol_p1, ol_p2;
    };
    static constexpr uint32_t kRecoveryMetaMagic = 0xE6AD0002u;
    RecoveryMeta* recovery_meta_ = nullptr;

public:
    explicit TpccDb(TpccConfig config);

    void loadInitialData() override;
    void generateTxns() override;
    void runBenchmark() override;

    void indexEpoch(uint32_t epoch_id);

    // Insert-CRID verifier for the 3 growing tables (NewOrder, Order,
    // OrderLine). Scans the CPU primary store at the insert range and
    // checks that every inserted CRID has a non-zero version. Mirrors
    // ycsb.cpp::verifyInsertedRecords. Gated on hybrid_staging mode +
    // config.verify_tpcc (--verify_tpcc CLI flag).
    void verifyInsertedRecords();

    // Verification-only negative-control hook. NO-OP unless
    // EPIC_RECOVERY_CORRUPT_ONE / EPIC_RECOVERY_SWAP_TWO is set. Perturbs the
    // final CPU Primary Store right before the end-of-run state hashes, to prove
    // the hash gate detects a corrupted recovery. Targets the stock table.
    void corruptPrimaryStoreForNegControl();

    // FNV-1a 64-bit hash over the in-use byte range of every
    // allocated CPU primary-store table (5 fixed + 3 growing), folded across
    // tables. Used by the replay-correctness check: a no-crash baseline and a
    // crash+replay run at the same seed must produce identical hashes. Only
    // meaningful with EPIC_TPCC_SEED set (deterministic runs).
    uint64_t hashCpuRecords();

    // Per-table positional + multiset (order-independent) hashes
    // for localizing non-determinism. If two runs match on a table's multiset
    // but differ on its positional hash, the divergence is record PLACEMENT
    // (same records, different CRIDs); if the multiset differs, it is record
    // CONTENT (and the table name pinpoints where). Gated on EPIC_TPCC_SEED;
    // logs one [TBL-HASH] line per table.
    void logCpuRecordsTableHashes();

    // Recovery-stable cumulative runtime insert counts for the 3 growing
    // tables, used to size their in-use range for hashing. Sourced from
    // the index's free-list cursors (restored across recovery), not the
    // per-stager counter (which a fresh recover process starts at 0).
    // See the .cpp.
    struct TpccGrowingCounts { uint64_t new_order = 0, order = 0, order_line = 0; };
    TpccGrowingCounts growingInsertCounts() const;

    // Advance the durable crash marker to `crash_epoch` and shift
    // the growing-table insert-cursor history (prev1<-current, prev2<-prev1).
    // Called once per epoch right after that epoch's writeback is queued/flushed,
    // so the marker reflects the in-flight writeback. No-op when recovery_meta_ is
    // null (non-durable runs stay byte-identical).
    void bumpRecoveryMarker(uint32_t crash_epoch);

    // Per-epoch pipeline extracted from runBenchmark's loop body so the
    // recover process can re-run an epoch after a GPU-crash restart. runEpoch
    // executes one epoch (index, admit, MVCC init, execute, flush) from the
    // host-resident input log (txn_array[epoch_id-1]).
    void runEpoch(uint32_t epoch_id);

    // Roll the Primary Store's dual-version tags back to end-of-(E-2)
    // before replaying E-1. Zeroes every version tag >= min_discard_epoch so the
    // surviving (<= E-2) slot is read as the base and the discarded (E-1/E) slot
    // becomes the write target on replay. growing_extent gives the in-use length
    // of the three growing tables at the crash point (cursors as of end-of-(E-1)).
    void rollbackPrimaryStoreVersions(uint32_t min_discard_epoch,
                                      const TpccFreeStarts& growing_extent);

    // GPU-state construction helpers, factored out of the ctor
    // (see tpcc.cpp).
    void initGpuPlannersAndSubmitter();
    void allocGpuRecordsAndVersions();
    void initGpuStagersAndWireDirty();

private:
};

} // namespace epic::tpcc

#endif // TPCC_H
