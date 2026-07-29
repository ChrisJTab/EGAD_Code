//
// Created by Shujian Qian on 2023-09-15.
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

// Hybrid staging: the per-table stager template; tpcc_hybrid_remap and
// tpcc_hybrid_executor provide the per-txn-type Remap and the GRID-addressed
// executor.
#include "benchmarks/tpcc_hybrid_stager.h"
#include "benchmarks/tpcc_hybrid_remap.h"
#include "benchmarks/tpcc_hybrid_executor.h"
#include "benchmarks/tpcc_staging_capacities.h"
#include "benchmarks/tpcc_workload_aware_capacities.h"

namespace epic::tpcc {
TpccTxnMix::TpccTxnMix(
    uint32_t new_order, uint32_t payment, uint32_t order_status, uint32_t delivery, uint32_t stock_level)
    : new_order(new_order)
    , payment(payment)
    , order_status(order_status)
    , delivery(delivery)
    , stock_level(stock_level)
{
    // Weights are interpreted as ratios summing to anything > 0. The standard
    // tpccfull mix sums to 100 (canonical percentages). The deck mix sums to
    // 23 (10 NewOrder + 10 Payment + 1 each of OrderStatus, Delivery,
    // StockLevel) per Clause 5.2.4.2 of the TPC-C spec. The weight sum
    // determines the txn_type_dist range in TpccTxnGenerator.
    assert(new_order + payment + order_status + delivery + stock_level > 0);
}

TpccDb::TpccDb(TpccConfig config)
    : config(config)
    , txn_array(config.epochs)
    , index_input(config.num_txns, config.index_device, false)
    , index_output(config.num_txns, config.index_device)
    , initialization_input(config.num_txns, config.initialize_device, false)
    , initialization_output(config.num_txns, config.initialize_device)
    , execution_param_input(config.num_txns, config.execution_device, false)
    , execution_plan_input(config.num_txns, config.execution_device, false)
    , cpu_aux_index(config)
    , packed_txn_array_builder(config.num_txns)
{
    //    index = std::make_shared<TpccCpuIndex>(config);
    // gpu_aux_index is constructed in the body (not the init list) so
    // the recovery rebuild can replace it with a fresh instance. See
    // tpcc.h's member declaration for the rationale.
    gpu_aux_index = std::make_unique<TpccGpuAuxIndex<TpccTxnArrayT, TpccTxnParamArrayT>>(config);
    for (int i = 0; i < config.epochs; ++i)
    {
        txn_array[i] = TpccTxnArrayT(config.num_txns, DeviceType::CPU);
    }
    if (config.index_device == DeviceType::CPU)
    {
        index = std::make_shared<TpccCpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>>(config);
    }
    else if (config.index_device == DeviceType::GPU)
    {
        // Construct the CPU shadow first so the GPU index can bind a
        // reference to it. The shadow owns the durable host-side copy
        // of the key->CRID mapping, with a lifetime independent of the
        // GPU index.
        const uint32_t maxO_ol = flatOLEnabled(config.execution_mode) ? computeOLMaxO(config) : 3000u;
        cpu_shadow_ = std::make_unique<TpccCpuShadowIndex>(config, maxO_ol);
        index = std::make_shared<TpccGpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>>(config, *cpu_shadow_);
    }
    else
    {
        throw std::runtime_error("Unsupported index device");
    }

    input_index_bridge.Link(txn_array[0], index_input);
    index_initialization_bridge.Link(index_output, initialization_input);
    index_execution_param_bridge.Link(index_output, execution_param_input);
    initialization_execution_plan_bridge.Link(initialization_output, execution_plan_input);

    if (config.initialize_device == DeviceType::GPU)
    {
        initGpuPlannersAndSubmitter();
    }
    else
    {
        auto &logger = Logger::GetInstance();
        logger.Error("Unsupported initialize device");
        exit(-1);
    }

    if (config.execution_device == DeviceType::GPU)
    {
        /* TODO: initialize records & versions */
        auto &logger = Logger::GetInstance();
        logger.Info("Allocating records and versions");

        GpuAllocator allocator;

        // HBM-aware autosizer for hybrid_staging mode. Sets cfg.cache_capacity_<X>
        // so the per-table GPU caches (allocated below as cacheCapacity<X>() *
        // sizeof(Record<X>)) fit within the available HBM budget. Skipped for
        // gpu_only (which stays at <X>TableSize() via the unset cap default,
        // i.e. cacheCapacity<X>() returning the table size).
        if (config.execution_mode == ExecMode::HYBRID_STAGING)
        {
            TpccRecordBytes record_bytes = {
                sizeof(Record<WarehouseValue>),
                sizeof(Record<DistrictValue>),
                sizeof(Record<CustomerValue>),
                sizeof(Record<ItemValue>),
                sizeof(Record<StockValue>),
                sizeof(Record<NewOrderValue>),
                sizeof(Record<OrderValue>),
                sizeof(Record<OrderLineValue>),
            };
            // Measure HBM via GpuAllocator (which wraps cudaMemGetInfo), so
            // tpcc.cpp doesn't need cuda_runtime.h.
            size_t free_bytes = 0, total_bytes = 0;
            allocator.GetMemoryInfo(free_bytes, total_bytes);
            // Workload-aware autosizer (tpcc_workload_aware_capacities.h):
            // derives per-table cache sizes from the TPC-C transaction-mix
            // access patterns rather than from available HBM. Safe for every
            // mix including NP (the derive step carries insert headroom for
            // the growing tables).
            computeAndApplyTpccStagingCapacitiesWorkloadAware(config, record_bytes, free_bytes, total_bytes);

            // Sync the autosizer's writes back to the
            // member. Both autosizers above mutate `cfg.cache_capacity_*`
            // (the three growing tables: NO/O/OL). `config` here is the
            // ctor parameter, which shadows the member of the same name.
            // The member-init `: config(config)` ran BEFORE the autosizer,
            // so without this write-back the member's cache_capacity_<X>
            // fields stay 0 (YcsbConfig default) and cacheCapacity<X>()
            // falls back to <X>TableSize(). At W=128 the OL fallback is
            // 57.6M records (~7 GB), which would overflow HBM on any
            // later reconstruction from the member (the autosizer caps
            // it at ~few GB here).
            this->config = config;
        }

        allocGpuRecordsAndVersions();

        // hybrid_staging path: allocate the per-table CPU primary store
        // and 8 TpccHybridStager instances. The existing GPU records
        // arrays serve as the cache pool. History gets no stager: the
        // executor neither reads nor writes it.
        if (config.execution_mode == ExecMode::HYBRID_STAGING)
        {
            logger.Info("Allocating CPU primary store and 8 TpccHybridStager instances");

            size_t w_cpu  = sizeof(Record<WarehouseValue>)  * config.warehouseTableSize();
            size_t d_cpu  = sizeof(Record<DistrictValue>)   * config.districtTableSize();
            size_t c_cpu  = sizeof(Record<CustomerValue>)   * config.customerTableSize();
            size_t i_cpu  = sizeof(Record<ItemValue>)       * config.itemTableSize();
            size_t s_cpu  = sizeof(Record<StockValue>)      * config.stockTableSize();
            size_t no_cpu = sizeof(Record<NewOrderValue>)   * config.newOrderTableSize();
            size_t o_cpu  = sizeof(Record<OrderValue>)      * config.orderTableSize();
            size_t ol_cpu = sizeof(Record<OrderLineValue>)  * config.orderLineTableSize();
            logger.Info("CPU primary store sizes: W={} D={} C={} I={} S={} NO={} O={} OL={}",
                formatSizeBytes(w_cpu), formatSizeBytes(d_cpu), formatSizeBytes(c_cpu),
                formatSizeBytes(i_cpu), formatSizeBytes(s_cpu), formatSizeBytes(no_cpu),
                formatSizeBytes(o_cpu), formatSizeBytes(ol_cpu));

            // In durable mode (EPIC_DURABLE_STORE/RECOVER_FROM,
            // recovery experiment only) the 8 primary-store tables are mmap-backed
            // so they survive a real GPU-fault process death; the recover process
            // re-maps them. OFF -> the EXACT existing allocation (Malloc for static,
            // Malloc/MallocPageable for growing), so throughput runs are byte-identical.
            const bool durable = (std::getenv("EPIC_DURABLE_STORE") != nullptr ||
                                  std::getenv("EPIC_RECOVER_FROM") != nullptr);
            cpu_records.warehouse_record  = static_cast<Record<WarehouseValue>*>( durable ? MallocDurable("tpcc_warehouse", w_cpu) : Malloc(w_cpu));
            cpu_records.district_record   = static_cast<Record<DistrictValue>*>(  durable ? MallocDurable("tpcc_district",  d_cpu) : Malloc(d_cpu));
            cpu_records.customer_record   = static_cast<Record<CustomerValue>*>(  durable ? MallocDurable("tpcc_customer",  c_cpu) : Malloc(c_cpu));
            cpu_records.item_record       = static_cast<Record<ItemValue>*>(      durable ? MallocDurable("tpcc_item",      i_cpu) : Malloc(i_cpu));
            cpu_records.stock_record      = static_cast<Record<StockValue>*>(     durable ? MallocDurable("tpcc_stock",     s_cpu) : Malloc(s_cpu));
            // The three growing-table primary stores (NewOrder, Order, OrderLine)
            // get a pageable allocation when EPIC_PAGEABLE_PRIMARY=1 (no upfront
            // page commit at deep E). In durable mode they are mmap-backed instead.
            const bool pageable_primary =
                envBoolOrHybridDefault("EPIC_PAGEABLE_PRIMARY", config.execution_mode);
            if (pageable_primary && !durable) {
                logger.Info("[PAGEABLE-PRIMARY] NewOrder, Order, OrderLine "
                            "primary stores allocated pageable (no upfront page commit, no memset)");
            }
            // Growing tables take the huge-page pageable variant: their
            // writeback CRIDs are near-sequential (insert allocation order),
            // so first-touch faults drop 512x with no meaningful commit
            // waste. Static update tables (stock, customer, ...) stay 4 KB —
            // their dirty sets are sparse-random and 2 MB granules would
            // balloon RSS.
            cpu_records.new_order_record  = static_cast<Record<NewOrderValue>*>(  durable ? MallocDurable("tpcc_new_order",  no_cpu, /*pin=*/false) : (pageable_primary ? MallocPageableHuge(no_cpu) : Malloc(no_cpu)));
            cpu_records.order_record      = static_cast<Record<OrderValue>*>(     durable ? MallocDurable("tpcc_order",      o_cpu,  /*pin=*/false) : (pageable_primary ? MallocPageableHuge(o_cpu)  : Malloc(o_cpu)));
            cpu_records.order_line_record = static_cast<Record<OrderLineValue>*>( durable ? MallocDurable("tpcc_order_line", ol_cpu, /*pin=*/false) : (pageable_primary ? MallocPageableHuge(ol_cpu): Malloc(ol_cpu)));

            // Deterministic-hash support. MallocPageable (above) skips
            // memset to preserve production's lazy page commit at deep E / large
            // W, which leaves the unwritten portions of the growing-table primary
            // stores uninitialized and would pollute hashCpuRecords. Under a
            // deterministic verification run (EPIC_TPCC_SEED set) we zero them so
            // the state hash sees a deterministic background plus the run's
            // deterministic writes. Production runs (no seed) keep the pageable
            // no-commit behavior untouched.
            // !durable: in durable/recover mode the growing
            // tables are MallocDurable (zeroed on create, PRESERVED on recover),
            // so this explicit zeroing must be skipped -- a recover run would
            // otherwise wipe the durable NO/O/OL data the crash left behind.
#ifdef EGAD_VALIDATION
            if (pageable_primary && !durable && std::getenv("EPIC_TPCC_SEED")) {
                std::memset(cpu_records.new_order_record,  0, no_cpu);
                std::memset(cpu_records.order_record,      0, o_cpu);
                std::memset(cpu_records.order_line_record, 0, ol_cpu);
                logger.Info("zeroed pageable growing-table primary stores for deterministic hashing");
            }
#endif // EGAD_VALIDATION

            initGpuStagersAndWireDirty();
        }

        /* TODO: execution input need to be transferred too, currently using placeholders */
//        executor =
//            std::make_shared<GpuExecutor>(records, versions, initialization_input, initialization_output, config);
        if (config.execution_mode == ExecMode::HYBRID_STAGING) {
            // hybrid_staging path: use the GRID-aware HybridExecutor. The
            // records/versions pointers are still the per-table arrays in
            // `records` / `versions` (the cache pool is the GPU records
            // array; the executor reads params->X_id, which is a GRID after
            // the per-txn-type Remap dispatched in runBenchmark).
            executor = std::make_shared<HybridExecutor<TpccTxnParamArrayT, TpccTxnExecPlanArrayT>>(
                records, versions, execution_param_input, execution_plan_input, config);
        } else {
            executor = std::make_shared<GpuExecutor<TpccTxnParamArrayT, TpccTxnExecPlanArrayT>>(
                records, versions, execution_param_input, execution_plan_input, config);
        }
    }
    else if (config.execution_device == DeviceType::CPU)
    {
        auto &logger = Logger::GetInstance();
        logger.Info("Allocating records and versions");

        /* CAUTION: version size is based on the number of transactions, and will cause sync issue if too small */
        size_t warehouse_rec_size = sizeof(Record<WarehouseValue>) * config.warehouseTableSize();
        size_t warehouse_ver_size = sizeof(Version<WarehouseValue>) * config.num_txns;
        logger.Info("Warehouse record: {}, version: {}", formatSizeBytes(warehouse_rec_size),
                    formatSizeBytes(warehouse_ver_size));
        records.warehouse_record = static_cast<Record<WarehouseValue> *>(Malloc(warehouse_rec_size));
        versions.warehouse_version = static_cast<Version<WarehouseValue> *>(Malloc(warehouse_ver_size));

        size_t district_rec_size = sizeof(Record<DistrictValue>) * config.districtTableSize();
        size_t district_ver_size = sizeof(Version<DistrictValue>) * config.num_txns;
        logger.Info(
            "District record: {}, version: {}", formatSizeBytes(district_rec_size), formatSizeBytes(district_ver_size));
        records.district_record = static_cast<Record<DistrictValue> *>(Malloc(district_rec_size));
        versions.district_version = static_cast<Version<DistrictValue> *>(Malloc(district_ver_size));

        size_t customer_rec_size = sizeof(Record<CustomerValue>) * config.customerTableSize();
        size_t customer_ver_size = sizeof(Version<CustomerValue>) * config.num_txns;
        logger.Info(
            "Customer record: {}, version: {}", formatSizeBytes(customer_rec_size), formatSizeBytes(customer_ver_size));
        records.customer_record = static_cast<Record<CustomerValue> *>(Malloc(customer_rec_size));
        versions.customer_version = static_cast<Version<CustomerValue> *>(Malloc(customer_ver_size));

        /* TODO: history table is too big */
        //        size_t history_rec_size = sizeof(Record<HistoryValue>) * config.historyTableSize();
        //        size_t history_ver_size = sizeof(Version<HistoryValue>) * config.historyTableSize();
        //        logger.Info("History record: {}, version: {}", formatSizeBytes(history_rec_size),
        //                    formatSizeBytes(history_ver_size));
        //        records.history_record = static_cast<Record<HistoryValue> *>(Malloc(history_rec_size));
        //        versions.history_version = static_cast<Version<HistoryValue> *>(Malloc(history_ver_size));

        size_t new_order_rec_size = sizeof(Record<NewOrderValue>) * config.newOrderTableSize();
        size_t new_order_ver_size = sizeof(Version<NewOrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("NewOrder record: {}, version: {}", formatSizeBytes(new_order_rec_size),
                    formatSizeBytes(new_order_ver_size));
        records.new_order_record = static_cast<Record<NewOrderValue> *>(Malloc(new_order_rec_size));
        versions.new_order_version = static_cast<Version<NewOrderValue> *>(Malloc(new_order_ver_size));

        size_t order_rec_size = sizeof(Record<OrderValue>) * config.orderTableSize();
        size_t order_ver_size = sizeof(Version<OrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("Order record: {}, version: {}", formatSizeBytes(order_rec_size), formatSizeBytes(order_ver_size));
        records.order_record = static_cast<Record<OrderValue> *>(Malloc(order_rec_size));
        versions.order_version = static_cast<Version<OrderValue> *>(Malloc(order_ver_size));

        size_t order_line_rec_size = sizeof(Record<OrderLineValue>) * config.orderLineTableSize();
        size_t order_line_ver_size = sizeof(Version<OrderLineValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("OrderLine record: {}, version: {}", formatSizeBytes(order_line_rec_size),
                    formatSizeBytes(order_line_ver_size));
        records.order_line_record = static_cast<Record<OrderLineValue> *>(Malloc(order_line_rec_size));
        versions.order_line_version = static_cast<Version<OrderLineValue> *>(Malloc(order_line_ver_size));

        size_t item_rec_size = sizeof(Record<ItemValue>) * config.itemTableSize();
        size_t item_ver_size = sizeof(Version<ItemValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("Item record: {}, version: {}", formatSizeBytes(item_rec_size), formatSizeBytes(item_ver_size));
        records.item_record = static_cast<Record<ItemValue> *>(Malloc(item_rec_size));
        versions.item_version = static_cast<Version<ItemValue> *>(Malloc(item_ver_size));

        size_t stock_rec_size = sizeof(Record<StockValue>) * config.stockTableSize();
        size_t stock_ver_size = sizeof(Version<StockValue>) * config.num_txns * 15;
        logger.Info("Stock record: {}, version: {}", formatSizeBytes(stock_rec_size), formatSizeBytes(stock_ver_size));
        records.stock_record = static_cast<Record<StockValue> *>(Malloc(stock_rec_size));
        versions.stock_version = static_cast<Version<StockValue> *>(Malloc(stock_ver_size));
        executor = std::make_shared<CpuExecutor<TpccTxnParamArrayT, TpccTxnExecPlanArrayT>>(
            records, versions, execution_param_input, execution_plan_input, config);
    }
    else
    {
        auto &logger = Logger::GetInstance();
        logger.Error("Unsupported initialize device");
        exit(-1);
    }
}


void TpccDb::generateTxns()
{
    auto &logger = Logger::GetInstance();

#ifdef EGAD_VALIDATION
    // Log the deterministic seed when set (the RNG sites read it via
    // tpccSeedOrEntropy at construction). Unset => fresh entropy, as before.
    if (const char *seed_env = std::getenv("EPIC_TPCC_SEED")) {
        logger.Info("EPIC_TPCC_SEED={} (deterministic txn + initial-data generation)", seed_env);
    }
#endif // EGAD_VALIDATION

    TpccTxnGenerator generator(config);
    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        logger.Info("Generating epoch {}", epoch);
        TpccTxnArrayT &txn_input_array = txn_array[epoch];
        uint32_t curr_size = 0;
        for (size_t i = 0; i < config.num_txns; ++i)
        {
#if 0
            BaseTxn *txn = txn_array[epoch].getTxn(i);
            uint32_t timestamp = epoch * config.num_txns + i;
            generator.generateTxn(txn, timestamp);
#else
            TpccTxnType txn_type = generator.getTxnType();
            constexpr uint32_t txn_sizes[6] = {0, BaseTxnSize<NewOrderTxnInput<FixedSizeTxn>>::value,
                BaseTxnSize<PaymentTxnInput>::value, BaseTxnSize<OrderStatusTxnInput>::value,
                BaseTxnSize<DeliveryTxnInput>::value, BaseTxnSize<StockLevelTxnInput>::value};
            txn_input_array.index[i] = curr_size;
            curr_size += txn_sizes[static_cast<uint32_t>(txn_type)];
            txn_input_array.size = curr_size;
            BaseTxn *txn = txn_input_array.getTxn(i);
            uint32_t timestamp = epoch * config.num_txns + i;
            generator.generateTxn(txn_type, txn, timestamp);
#endif
        }
    }
}

void TpccDb::loadInitialData()
{
    // Load the CPU shadow before the GPU index.
    if (cpu_shadow_) cpu_shadow_->loadInitialData();
    index->loadInitialData();
    // cpu_aux_index.loadInitialData(); /* cpu aux index replace by gpu aux index */
    gpu_aux_index->loadInitialData();
}


// Within-epoch crash injector (copied from ycsb.cpp). Fires a
// REAL GPU fault at a chosen (EPIC_CRASH_AT_EPOCH, EPIC_CRASH_AT_PHASE): phase 0
// = top of epoch (before runEpoch); 1 = after admission/staging; 2 = after
// execution; 3 = after E's writeback is queued (marker E+1, bytes landing);
// 4 = E-1 fully drained, before the marker bump (marker E, none of E durable);
// 5 = after the bump, before E's writeback starts (marker E+1, none of E
// durable). Covering distinct durable states
// models the untimed-ness of a real fault. Never fires in recover-mode.
static void maybeCrashAt(uint32_t epoch_id, int phase)
{
#ifdef EGAD_VALIDATION
    static const int  kCrashEpoch = []{ const char* s = std::getenv("EPIC_CRASH_AT_EPOCH"); return s ? std::atoi(s) : -1; }();
    static const int  kCrashPhase = []{ const char* s = std::getenv("EPIC_CRASH_AT_PHASE"); return s ? std::atoi(s) : 0; }();
    static const bool kRecover    = std::getenv("EPIC_RECOVER_FROM") != nullptr;
    if (!kRecover && kCrashEpoch == static_cast<int>(epoch_id) && kCrashPhase == phase) {
        Logger::GetInstance().Info("[CRASH] REAL GPU fault at epoch {} phase {} -- process will die", epoch_id, phase);
        injectGpuFault();  // does not return
    }
#else
    (void)epoch_id; (void)phase;
#endif
}

// Per-epoch pipeline, shared by the normal run loop and the recovery
// replay path. The 8 FlushHandles are TpccDb members, so no cross-epoch
// state needs threading through a parameter.
void TpccDb::runEpoch(uint32_t epoch_id)
{
    auto &logger = Logger::GetInstance();
    std::chrono::high_resolution_clock::time_point start_time, end_time;

        /* cpu aux index */
        {
            start_time = std::chrono::high_resolution_clock::now();

            uint32_t index_epoch_id = epoch_id - 1;
            /* cpu_aux_index replaced by gpu_aux_index */
            // cpu_aux_index.insertTxnUpdates(txn_array[index_epoch_id], epoch_id);
            // cpu_aux_index.performRangeQueries(txn_array[index_epoch_id], epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} cpu aux index time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            input_index_bridge.Link(txn_array[index_epoch_id], index_input);
            input_index_bridge.StartTransfer();
            input_index_bridge.FinishTransfer();
            packed_txn_array_builder.buildPackedTxnArrayGpu(index_input, index_output);
            packed_txn_array_builder.buildPackedTxnArrayGpu(index_input, initialization_output);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} index_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* gpu aux index */
        {

            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            gpu_aux_index->insertTxnUpdates(index_input, index_epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} gpu aux index time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

            start_time = std::chrono::high_resolution_clock::now();
            gpu_aux_index->performRangeQueries(index_input, index_output, index_epoch_id);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} gpu aux index part2 time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* index */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            index->indexTxns(index_input, index_output, index_epoch_id);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} indexing time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            index_initialization_bridge.StartTransfer();
            index_initialization_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} init_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

        }

        /* submit */
        {
            start_time = std::chrono::high_resolution_clock::now();
            submitter->submit(initialization_input);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} submission time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* hybrid_staging: per-table prepareEpoch */
        // Each stager consumes the per-table d_all/read/write/insert_keys
        // arrays the submitter built above. n_all is the bound; the read /
        // write / insert subsets are sentinel-padded parallel to all_keys, so
        // their counts are <= n_all (the submitter's countReadWriteInsertKeys
        // pass populated curr_num_read_ops / curr_num_write_ops /
        // curr_num_insert_ops on each planner).
        if (config.execution_mode == ExecMode::HYBRID_STAGING)
        {
            start_time = std::chrono::high_resolution_clock::now();
            // [INV-TL] Per-stager timestamps so the parser can build a
            // per-epoch timeline of which stager's prepareEpoch was running
            // when. The stager name is logged before/after the call; the
            // line's own timestamp is the boundary.
            auto run = [&](const char* name, auto& stager, auto& planner, FlushHandle& flush) {
                FlushHandle* fh = (config.overlap_flush && flush.valid) ? &flush : nullptr;
                logger.Info("[INV-TL] epoch={} stager={} phase=prepareEpoch event=start", epoch_id, name);
                stager->prepareEpoch(epoch_id,
                    planner->d_all_keys,    planner->curr_num_ops,
                    planner->d_read_keys,   planner->curr_num_read_ops,
                    planner->d_write_keys,  planner->curr_num_write_ops,
                    planner->d_insert_keys, planner->curr_num_insert_ops,
                    fh);
                logger.Info("[INV-TL] epoch={} stager={} phase=prepareEpoch event=end", epoch_id, name);
            };
            // Flag this epoch's deleted NewOrder rows' cache slots
            // reclaim-first before eviction runs (single-threaded region,
            // before the parallel sections). Safe within the deleting
            // epoch: any slot the epoch still touches is needed-protected,
            // and a slot with an in-flight writeback is pinned.
            if (config.txn_mix.delivery > 0) {
                if (auto* gi = dynamic_cast<TpccGpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>*>(index.get())) {
                    new_order_stager->mark_reclaimable(gi->noDeleteCridsDevice(), gi->numNoDeletesThisEpoch());
                }
            }
            // The 8 stagers prepare concurrently, each on its own
            // prep_stream_ (created in the TpccHybridStager constructor).
            static const bool kPrepEpochParallel = true;
            if (kPrepEpochParallel) {
                // 8 stagers run prepareEpoch concurrently, each on its own
                // prep_stream_. This is stable at num_threads(8) only because
                // the helpers issue no synchronous cudaMemcpy: a synchronous
                // copy is a global cross-stream sync point, and 8 threads
                // concurrently issuing them corrupts CUB output (bulk_lookup
                // returns sentinels for valid CRIDs). Keep helper transfers
                // cudaMemsetAsync / cudaMemcpyAsync on the per-stager stream.
                #pragma omp parallel sections num_threads(8)
                {
                    #pragma omp section
                    run("warehouse",  warehouse_stager,  warehouse_planner,  warehouse_flush);
                    #pragma omp section
                    run("district",   district_stager,   district_planner,   district_flush);
                    #pragma omp section
                    run("customer",   customer_stager,   customer_planner,   customer_flush);
                    #pragma omp section
                    run("item",       item_stager,       item_planner,       item_flush);
                    #pragma omp section
                    run("stock",      stock_stager,      stock_planner,      stock_flush);
                    #pragma omp section
                    run("new_order",  new_order_stager,  new_order_planner,  new_order_flush);
                    #pragma omp section
                    run("order",      order_stager,      order_planner,      order_flush);
                    #pragma omp section
                    run("order_line", order_line_stager, order_line_planner, order_line_flush);
                }
            } else {
                run("warehouse",  warehouse_stager,  warehouse_planner,  warehouse_flush);
                run("district",   district_stager,   district_planner,   district_flush);
                run("customer",   customer_stager,   customer_planner,   customer_flush);
                run("item",       item_stager,       item_planner,       item_flush);
                run("stock",      stock_stager,      stock_planner,      stock_flush);
                run("new_order",  new_order_stager,  new_order_planner,  new_order_flush);
                run("order",      order_stager,      order_planner,      order_flush);
                run("order_line", order_line_stager, order_line_planner, order_line_flush);
            }
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} hybrid_staging:prepareEpoch time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        // crash point 1: after admission/staging (records in
        // cache, E-1's writeback still in flight, E not yet written back).
        maybeCrashAt(epoch_id, 1);

        /* initialize */
        {
            start_time = std::chrono::high_resolution_clock::now();

            warehouse_planner->InitializeExecutionPlan();
            district_planner->InitializeExecutionPlan();
            customer_planner->InitializeExecutionPlan();
            history_planner->InitializeExecutionPlan();
            new_order_planner->InitializeExecutionPlan();
            order_planner->InitializeExecutionPlan();
            order_line_planner->InitializeExecutionPlan();
            item_planner->InitializeExecutionPlan();
            stock_planner->InitializeExecutionPlan();

            warehouse_planner->FinishInitialization();
            district_planner->FinishInitialization();
            customer_planner->FinishInitialization();
            history_planner->FinishInitialization();
            new_order_planner->FinishInitialization();
            order_planner->FinishInitialization();
            order_line_planner->FinishInitialization();
            item_planner->FinishInitialization();
            stock_planner->FinishInitialization();

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} initialization time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            index_execution_param_bridge.StartTransfer();
            initialization_execution_plan_bridge.StartTransfer();
            index_execution_param_bridge.FinishTransfer();
            initialization_execution_plan_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} exec_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* hybrid_staging: per-txn-type Remap (CRID → GRID) on execution_param_input */
        // Each per-txn-type kernel rewrites every record_id field in its
        // TxnParams via the appropriate per-table CridGridIndex flat lookup.
        // This must happen AFTER execution_param_input has the latest copy
        // of the indexer's CRIDs (post-bridge) and BEFORE the executor reads
        // params->X_id (which now resolves to a GRID into the cache pool).
        if (config.execution_mode == ExecMode::HYBRID_STAGING)
        {
            start_time = std::chrono::high_resolution_clock::now();
            TpccCridGridViews views{
                .warehouse  = warehouse_stager ->crid_to_grid_index().device_view(),
                .district   = district_stager  ->crid_to_grid_index().device_view(),
                .customer   = customer_stager  ->crid_to_grid_index().device_view(),
                .item       = item_stager      ->crid_to_grid_index().device_view(),
                .stock      = stock_stager     ->crid_to_grid_index().device_view(),
                .new_order  = new_order_stager ->crid_to_grid_index().device_view(),
                .order      = order_stager     ->crid_to_grid_index().device_view(),
                .order_line = order_line_stager->crid_to_grid_index().device_view(),
            };
            launchRemapTpccTxns(execution_param_input, config.num_txns, views);
            // launchRemapTpccTxns uses gpu_err_check internally; the kernel
            // is launched on the default stream so the next stream-0 op
            // (the executor kernel) implicitly waits for it.
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} hybrid_staging:remap time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* execution */
        {
            start_time = std::chrono::high_resolution_clock::now();
            executor->execute(epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} execution time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        // crash point 2: after execution (E's writes are in the
        // GPU cache, NOT yet written back; E-1's writeback still in flight).
        maybeCrashAt(epoch_id, 2);

        /* hybrid_staging: per-table flush */
        if (config.execution_mode == ExecMode::HYBRID_STAGING)
        {
            if (config.overlap_flush)
            {
                auto sf = [&](const char* name, auto& stager, FlushHandle& flush) {
                    logger.Info("[INV-TL] epoch={} stager={} phase=sync_flush event=start", epoch_id, name);
                    stager->sync_flush(flush);
                    logger.Info("[INV-TL] epoch={} stager={} phase=sync_flush event=end", epoch_id, name);
                };
                auto sa = [&](const char* name, auto& stager, FlushHandle& flush) {
                    logger.Info("[INV-TL] epoch={} stager={} phase=start_flush_async event=start", epoch_id, name);
                    stager->start_flush_epoch_async(epoch_id, flush);
                    // Also emit the spawned worker's std::thread::id (if it's
                    // joinable, i.e. spawning succeeded). This gives the parser
                    // a deterministic stager <-> worker-thread mapping
                    // independent of worker: log ordering.
                    std::ostringstream tid_s;
                    if (flush.worker.joinable()) tid_s << flush.worker.get_id();
                    else tid_s << "(no-spawn)";
                    logger.Info("[INV-TL] epoch={} stager={} phase=start_flush_async event=end worker_tid={} valid={} stager_addr={}",
                        epoch_id, name, tid_s.str(), flush.valid ? 1 : 0, (void*)stager.get());
                };
                // The marker/writeback ordering constraint exists only when
                // a durable marker exists: with recovery_meta_ set, no byte
                // of epoch E may reach the Primary Store before
                // bumpRecoveryMarker(E+1) below, so the worker submits are
                // deferred past the bump. Non-durable runs have nothing to
                // recover and keep the in-section submit: deferring there
                // shifts the 8 workers' D2H+scatter into the next epoch's
                // index_transfer window and costs ~5% at deck W=128.
                const bool defer_submit = (recovery_meta_ != nullptr);
                {
                    // Fused: each stager drains its own E-1 writeback
                    // (sync_flush) and immediately prepares E's flush
                    // (start_flush_epoch_async: collect, sort, pack, pin),
                    // all 8 stagers concurrently — same OMP-sections
                    // dispatch and per-stager-stream isolation contract as
                    // prepareEpoch above. A light stager's flush preparation
                    // overlaps the heavy OL worker's residual drain instead
                    // of queueing behind it in the serial loops; per-stager
                    // (drain → prepare) ordering is preserved inside each
                    // section. In durable mode the flush workers are
                    // submitted below, after the recovery marker advances:
                    // submitted inside the block, a light stager's writeback
                    // could land epoch-E bytes while the durable marker
                    // still reads E, and a crash there would roll back to an
                    // end-of-(E-2) state whose base values E's writeback had
                    // already overwritten (both slots of a record updated in
                    // E-1 and E would carry tags >= E-1, so demotion
                    // destroys the only surviving base).
                    auto fs = [&](const char* name, auto& stager, FlushHandle& flush) {
                        sf(name, stager, flush);
                        sa(name, stager, flush);
                        if (!defer_submit) stager->submit_flush_worker(flush);
                    };
                    start_time = std::chrono::high_resolution_clock::now();
                    // order_line's section leads: its drain and flush
                    // preparation are the heaviest, so dispatching it first
                    // keeps it off the block's critical-path tail.
                    #pragma omp parallel sections num_threads(8)
                    {
                        #pragma omp section
                        fs("order_line", order_line_stager, order_line_flush);
                        #pragma omp section
                        fs("warehouse",  warehouse_stager,  warehouse_flush);
                        #pragma omp section
                        fs("district",   district_stager,   district_flush);
                        #pragma omp section
                        fs("customer",   customer_stager,   customer_flush);
                        #pragma omp section
                        fs("item",       item_stager,       item_flush);
                        #pragma omp section
                        fs("stock",      stock_stager,      stock_flush);
                        #pragma omp section
                        fs("new_order",  new_order_stager,  new_order_flush);
                        #pragma omp section
                        fs("order",      order_stager,      order_flush);
                    }
                    end_time = std::chrono::high_resolution_clock::now();
                    // The drain no longer exists as a separate serial phase;
                    // report 0 for it and the fused region's wall time under
                    // flush_start_async so timing parsers that sum the two
                    // lines keep working.
                    logger.Info("Epoch {} hybrid_staging:flush_sync_prev time: {} us", epoch_id, 0);
                    logger.Info("Epoch {} hybrid_staging:flush_start_async time: {} us", epoch_id,
                        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
                }

                // crash point 4 (durable mode): every stager has drained
                // E-1 and prepared E's flush; no worker submitted, marker
                // still E. Recover rolls back to end-of-(E-2) and replays
                // E-1, E -- sound because no byte of E has reached the
                // Primary Store.
                maybeCrashAt(epoch_id, 4);

                // Advance the crash marker to E+1 at the one point where
                // both directions are safe: every stager's sync_flush has
                // retired E-1 (end-of-(E-1) fully durable), and E's
                // writeback has not been submitted (no end-of-(E-2) slot
                // overwritten yet). Earlier, E-1 could still be landing and
                // rolling back to end-of-(E-1) would read stale bases;
                // later, E's scatter is already overwriting end-of-(E-2)
                // slots (the writeback lands in the older slot) and rolling
                // back to end-of-(E-2) reads destroyed bases.
                bumpRecoveryMarker(epoch_id + 1);

                // crash point 5 (durable mode): marker E+1, E's writeback
                // still not submitted. Recover rolls back to end-of-(E-1),
                // replays E.
                maybeCrashAt(epoch_id, 5);

                if (defer_submit) {
                    // Launch the 8 flush workers, order_line first (heaviest
                    // scatter; the earliest submit masks its CV-wakeup
                    // latency). Every worker's first Primary Store byte
                    // lands strictly after the marker bump above (submit's
                    // mutex orders the marker stores before the worker's
                    // scatter).
                    order_line_stager->submit_flush_worker(order_line_flush);
                    warehouse_stager ->submit_flush_worker(warehouse_flush);
                    district_stager  ->submit_flush_worker(district_flush);
                    customer_stager  ->submit_flush_worker(customer_flush);
                    item_stager      ->submit_flush_worker(item_flush);
                    stock_stager     ->submit_flush_worker(stock_flush);
                    new_order_stager ->submit_flush_worker(new_order_flush);
                    order_stager     ->submit_flush_worker(order_flush);
                }

                // crash point 3: E's writeback queued + landing, marker E+1.
                maybeCrashAt(epoch_id, 3);
            }
            else
            {
                // crash point 4 (sync path): marker still E, E's writeback
                // not started. Recover rolls back to end-of-(E-2), replays
                // E-1, E.
                maybeCrashAt(epoch_id, 4);
                // Advance the marker BEFORE the writeback: E-1 is fully
                // durable (last epoch's periodicFlush completed
                // synchronously) and E has not started, so this is the
                // sync-path analog of the async bump point. The whole flush
                // below then runs under marker E+1, and a crash anywhere
                // inside it rolls back to end-of-(E-1) and replays E. Bumped
                // after the flush instead, the entire synchronous writeback
                // would land under a marker that still reads E, and recover
                // would roll back to an end-of-(E-2) state whose bases E's
                // writeback had already overwritten.
                bumpRecoveryMarker(epoch_id + 1);
                // crash point 5 (sync path): marker E+1, E's writeback not
                // started. Recover rolls back to end-of-(E-1), replays E.
                maybeCrashAt(epoch_id, 5);
                start_time = std::chrono::high_resolution_clock::now();
                warehouse_stager ->periodicFlush(epoch_id);
                district_stager  ->periodicFlush(epoch_id);
                customer_stager  ->periodicFlush(epoch_id);
                item_stager      ->periodicFlush(epoch_id);
                stock_stager     ->periodicFlush(epoch_id);
                new_order_stager ->periodicFlush(epoch_id);
                order_stager     ->periodicFlush(epoch_id);
                order_line_stager->periodicFlush(epoch_id);
                end_time = std::chrono::high_resolution_clock::now();
                logger.Info("Epoch {} hybrid_staging:periodicFlush time: {} us", epoch_id,
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
                // crash point 3 (sync path): epoch E fully written back,
                // marker E+1.
                maybeCrashAt(epoch_id, 3);
            }

            // [INV] Investigation instrumentation: cumulative per-stager PCIe
            // bytes after this epoch's flush. Same emit point in both sync
            // and async paths so async-vs-sync deltas can be reconstructed
            // by the parser. Stager order matches the dispatch order above
            // (warehouse, district, customer, item, stock, new_order, order,
            // order_line). One line per epoch keeps logs grep-able.
            logger.Info("[INV] epoch={} writeback_d2h_bytes_cum=W:{} D:{} C:{} I:{} S:{} N:{} O:{} L:{}",
                epoch_id,
                warehouse_stager  ->pcie_writeback_d2h_bytes(),
                district_stager   ->pcie_writeback_d2h_bytes(),
                customer_stager   ->pcie_writeback_d2h_bytes(),
                item_stager       ->pcie_writeback_d2h_bytes(),
                stock_stager      ->pcie_writeback_d2h_bytes(),
                new_order_stager  ->pcie_writeback_d2h_bytes(),
                order_stager      ->pcie_writeback_d2h_bytes(),
                order_line_stager ->pcie_writeback_d2h_bytes());
            logger.Info("[INV] epoch={} admit_h2d_bytes_cum=W:{} D:{} C:{} I:{} S:{} N:{} O:{} L:{}",
                epoch_id,
                warehouse_stager  ->pcie_admit_h2d_bytes(),
                district_stager   ->pcie_admit_h2d_bytes(),
                customer_stager   ->pcie_admit_h2d_bytes(),
                item_stager       ->pcie_admit_h2d_bytes(),
                stock_stager      ->pcie_admit_h2d_bytes(),
                new_order_stager  ->pcie_admit_h2d_bytes(),
                order_stager      ->pcie_admit_h2d_bytes(),
                order_line_stager ->pcie_admit_h2d_bytes());
        }
}


void TpccDb::runBenchmark()
{
    auto &logger = Logger::GetInstance();
    std::chrono::high_resolution_clock::time_point start_time, end_time;

    // [AUX] One-shot host-side guard: inspect the generated txn_array and
    // confirm no NewOrder o_id exceeds the GPU aux-index's
    // num_slots_per_district cap. The cap comes from the shared
    // TpccConfig::auxNumSlotsPerDistrict() (the same value the gpu_aux_index
    // ctor sizes its arrays to), so this guard tracks the array exactly. The
    // scan is cheap (~0.1s for 30M txns) and crashes from o_id overflow are
    // silent on the GPU side (illegal-memory-access caught only at the next
    // sync), so this turns a hard-to-trace runtime crash into a single startup
    // warning. Quiet unless something is wrong.
    if (config.execution_mode == ExecMode::HYBRID_STAGING)
    {
        const uint32_t cap = config.auxNumSlotsPerDistrict();
        uint32_t global_max_oid = 0;
        uint32_t first_overflow_epoch = 0;
        for (uint32_t e = 0; e < config.epochs; ++e) {
            uint32_t epoch_max = 0;
            for (uint32_t i = 0; i < config.num_txns; ++i) {
                BaseTxn* txn = txn_array[e].getTxn(i);
                if (txn->txn_type == static_cast<uint32_t>(TpccTxnType::NEW_ORDER)) {
                    auto* no = reinterpret_cast<NewOrderTxnInput<FixedSizeTxn>*>(txn->data);
                    if (no->o_id > epoch_max) epoch_max = no->o_id;
                    if (no->o_id > global_max_oid) global_max_oid = no->o_id;
                }
            }
            if (first_overflow_epoch == 0 && epoch_max >= cap) first_overflow_epoch = e + 1;
        }
        if (global_max_oid >= cap) {
            logger.Info("[AUX] *** WARNING: NewOrder o_id ({}) reaches GPU aux-index cap "
                "({}) at epoch {}. The kernel will likely fault with "
                "an illegal-memory-access. Bump num_slots_per_district in "
                "tpcc_gpu_aux_index.cu's ctor.",
                global_max_oid, cap, first_overflow_epoch);
        }
    }

    // Optional one-shot pre-warm: bring the OL records of the spec's
    // initial undelivered backlog onto the GPU before epoch 1 starts.
    // Removes the warmup hump where Delivery would otherwise miss for
    // the first ~25 epochs while it works through the initial backlog.
    // Skip the OL pre-warm in recover-mode. The pre-warm admits
    // the full INITIAL undelivered OL backlog to the GPU cache before the recover
    // block (below) rebuilds the cache index from the shadow, orphaning those
    // cache entries -- which corrupts OL recovery when rolling back near genesis
    // (the initial backlog is exactly what early-epoch Delivery touches). The
    // OL pre-warm runs once before epoch 1. It is a steady-state warmup only
    // (a no-pre-warm run is value-identical), so recovery replay's admission
    // still brings in every OL it needs; only the recover process
    // (EPIC_RECOVER_FROM) skips it.
    if (config.execution_mode == ExecMode::HYBRID_STAGING
        && order_line_stager
        && std::getenv("EPIC_RECOVER_FROM") == nullptr) {
        logger.Info("running OL pre-warm before epoch 1 (canonical default)");
        order_line_stager->prewarm_initial_undelivered_orderline(config.num_warehouses);
    }

    const auto run_start = std::chrono::high_resolution_clock::now();

    // Durable recovery metadata (member recovery_meta_; the
    // RecoveryMeta type + magic live in tpcc.h). Gated on a durable store
    // (EPIC_DURABLE_STORE / EPIC_RECOVER_FROM), so non-durable runs allocate
    // nothing and stay byte-identical. The crash marker is bumped per-epoch in
    // runEpoch once every stager has drained E-1 and before E's writeback is
    // submitted (bumpRecoveryMarker), so it reflects the durable writeback
    // frontier rather than the loop index.
    const bool kDurable = std::getenv("EPIC_DURABLE_STORE") || std::getenv("EPIC_RECOVER_FROM");
    if (kDurable && !recovery_meta_) {
        recovery_meta_ = static_cast<RecoveryMeta*>(MallocDurable("tpcc_meta", sizeof(RecoveryMeta)));
    }

    // Recover-mode (fresh process after a real GPU fault, gate
    // EPIC_RECOVER_FROM). The ctor already re-mapped the durable Primary Store,
    // and main.cpp's loadInitialData()/generateTxns() rebuilt the initial-pop
    // shadow + regenerated the input log. Here we roll the Primary Store back to
    // end-of-(E-2), rebuild the GPU indexes to f_{E-2}, and resume the loop at
    // E-1 so E-1 and E replay (the two-epoch recovery window). A fresh process
    // already has a clean CUDA context, so no in-process GPU reset is needed.
    const bool kRecover = std::getenv("EPIC_RECOVER_FROM") != nullptr;
    uint32_t start_epoch = 1;
    if (kRecover) {
        uint32_t E = 0;
        const TpccRecoveryCursors* cursors = nullptr;
        if (recovery_meta_) recovery_meta_->read(E, cursors);
        if (E < 2) {
            logger.Warn("[RECOVER] crash marker E={} < 2 (no end-of-(E-2) to roll back to); "
                        "falling back to full replay from epoch 1", E);
        } else {
            const TpccFreeStarts f_e2{ cursors->no_p2, cursors->o_p2, cursors->ol_p2 };  // f_{E-2}
            const uint32_t no_d_e2 = cursors->no_d_p2;  // NO delete cursor at end-of-(E-2)
            logger.Info("[RECOVER] crash marker E={}; roll back to end-of-({}) "
                        "(f_(E-2): NO={} O={} OL={}), resume at epoch {}",
                        E, static_cast<int>(E) - 2,
                        f_e2.new_order, f_e2.order, f_e2.order_line, E - 1);

            // 1) Demote the Primary Store's dual-version tags to end-of-(E-2).
            //    Scan the FULL growing-table headroom (not f_{E-2}) so E-1/E
            //    inserts from ANY crash phase get their >= E-1 tags demoted;
            //    unwritten slots carry tag 0 and are skipped. Matches YCSB's
            //    full-array scan.
            const uint64_t no_init = static_cast<uint64_t>(config.num_warehouses) * 10ull * 900ull;
            const uint64_t o_init  = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull;
            const uint64_t ol_init = static_cast<uint64_t>(config.num_warehouses) * 10ull * 3000ull * 15ull;
            const TpccFreeStarts full_extent{
                static_cast<uint32_t>(static_cast<uint64_t>(config.newOrderTableSize())  - no_init),
                static_cast<uint32_t>(static_cast<uint64_t>(config.orderTableSize())     - o_init),
                static_cast<uint32_t>(static_cast<uint64_t>(config.orderLineTableSize()) - ol_init)};
            // 1) Demote the Primary Store's dual-version tags to end-of-(E-2).
            rollbackPrimaryStoreVersions(E - 1, full_extent);
            // 2) Reconstruct the runtime insert shadows from the durable key
            //    arrays (layered on the initial-pop shadow loadInitialData
            //    built), then apply the NO delete log through d_(E-2).
            //    Deletes from E-1 and E are beyond the cursor, so their NO
            //    rows come back live and replay re-erases them.
            cpu_shadow_->reconstructInsertsFromDurable(f_e2);
            cpu_shadow_->applyNoDeletesFromDurable(no_d_e2);
#ifdef EGAD_VALIDATION
            // Liveness gate, reconstruction side: the NO shadow must equal
            // an independent replay of the logs to the same cursors. A lost
            // or extra delete application changes no store bytes, so only
            // this check can see it.
            if (config.txn_mix.delivery > 0) {
                cpu_shadow_->verifyNoLiveAgainstLogs(f_e2.new_order, no_d_e2);
            }
#endif // EGAD_VALIDATION
            // 3) Rebuild the aux index from the shadow. Recreate it FRESH first:
            //    main.cpp's loadInitialData() already built the aux, and
            //    rebuildFromShadow calls loadInitialData() internally, so reusing
            //    the loaded instance would double-load it. The old instance's
            //    device buffers are explicitly freed first (freeDeviceBuffers),
            //    so replacing it does not leak (there is no cudaFree dtor).
            gpu_aux_index->freeDeviceBuffers();
            gpu_aux_index = std::make_unique<TpccGpuAuxIndex<TpccTxnArrayT, TpccTxnParamArrayT>>(config);
            gpu_aux_index->rebuildFromShadow(*cpu_shadow_,
                                             cpu_records.order_record,
                                             cpu_records.order_line_record);
            // 4) Rebuild the primary index/cache from the shadow.
            if (auto* gi = dynamic_cast<TpccGpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>*>(index.get())) {
                gi->rebuildIndexesFromShadow(f_e2, no_d_e2);
            }
            logger.Info("[RECOVER] rebuild complete; resuming at epoch {}", E - 1);
            start_epoch = E - 1;
        }
    }

    for (uint32_t epoch_id = start_epoch; epoch_id <= config.epochs; ++epoch_id)            // Loop over all epochs
    {
        logger.Info("Running epoch {}", epoch_id);

        // Crash point 0: top of epoch (before the pipeline runs;
        // E-1's writeback in-flight, E not started). The crash marker is whatever
        // the last bump set it to (= E here, set at the end of runEpoch(E-1)
        // just before E-1's writeback was submitted), so recover rolls back to
        // end-of-(E-2). Original run only.
        maybeCrashAt(epoch_id, 0);

        runEpoch(epoch_id);
    }

    // End-of-run drain: after the last epoch's start_flush_epoch_async
    // calls, 8 worker threads may still be writing to CPU primary store.
    // Each sync_flush waits on its stager's persistent worker. Mirrors
    // ycsb.cpp's final-flush block but fanned out 8x.
    if (config.execution_mode == ExecMode::HYBRID_STAGING && config.overlap_flush)
    {
        const auto fstart = std::chrono::high_resolution_clock::now();
        warehouse_stager  ->sync_flush(warehouse_flush);
        district_stager   ->sync_flush(district_flush);
        customer_stager   ->sync_flush(customer_flush);
        item_stager       ->sync_flush(item_flush);
        stock_stager      ->sync_flush(stock_flush);
        new_order_stager  ->sync_flush(new_order_flush);
        order_stager      ->sync_flush(order_flush);
        order_line_stager ->sync_flush(order_line_flush);
        const auto fend = std::chrono::high_resolution_clock::now();
        logger.Info("Final flush_sync_prev time: {} us",
            std::chrono::duration_cast<std::chrono::microseconds>(fend - fstart).count());
    }

    // Aggregate end-of-run throughput line — mirrors ycsb.cpp's [RUN]
    // output. Harness scripts grep for this; without it, there is no
    // programmatic way to read TPC-C's throughput out of a run.
    const auto run_end = std::chrono::high_resolution_clock::now();
    const double run_seconds = std::chrono::duration<double>(run_end - run_start).count();
    const uint64_t total_txns = static_cast<uint64_t>(config.epochs) * config.num_txns;
    if (run_seconds > 0.0) {
        logger.Info("[RUN] {} epochs x {} txns = {} txns in {:.3f} s -> {:.0f} txns/s",
                    config.epochs, config.num_txns, total_txns,
                    run_seconds, total_txns / run_seconds);
    }

#ifdef EGAD_VALIDATION
    // State hash for replay-correctness verification, gated on a deterministic
    // run (EPIC_TPCC_SEED set). A no-crash baseline and a crash+replay run at the
    // same seed print identical hashes. Validation build only.
    if (std::getenv("EPIC_TPCC_SEED")) {
        corruptPrimaryStoreForNegControl();   // no-op unless neg-control env set
        logger.Info("[STATE-HASH] tpcc cpu_records FNV-1a 64-bit = 0x{:016x}", hashCpuRecords());
        logCpuRecordsTableHashes();
        // Liveness gate, end-of-run side: digest of the live NewOrder
        // key->CRID mapping derived from the durable logs. Deletes write
        // no record bytes, so the store hashes above cannot distinguish a
        // live NO row from a delivered one; this line can. Durable runs
        // only (the logs are the source).
        if (config.txn_mix.delivery > 0 && cpu_shadow_
            && (std::getenv("EPIC_DURABLE_STORE") || std::getenv("EPIC_RECOVER_FROM"))) {
            if (auto* gi = dynamic_cast<TpccGpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>*>(index.get())) {
                logger.Info("[STATE-HASH-LIVE] tpcc NO live-mapping = 0x{:016x}",
                            cpu_shadow_->noLiveDigestFromLogs(gi->getInsertCounts().new_order,
                                                              gi->getNoDeleteCount()));
            }
        }
    }
#endif // EGAD_VALIDATION

    // Insert verifier — gated on --verify_tpcc, only runs in hybrid_staging
    // mode. Walks the CPU primary store at insert ranges and confirms every
    // minted CRID has a non-zero version (which means the executor's INSERT
    // write reached CPU via the dirty-bit + flush pipeline). Catches the
    // failure class where the dirty bit lands on the wrong slot and the
    // executor's writes never reach CPU.
    if (config.verify_tpcc) {
        verifyInsertedRecords();
    }
}


void TpccDb::indexEpoch(uint32_t epoch_id)
{
    /* TODO: remove */
    auto &logger = Logger::GetInstance();
    logger.Error("Deprecated function");
    exit(-1);

    //    /* zero-indexed */
    //    uint32_t index_epoch_id = epoch_id - 1;
    //
    //    /* it's important to index writes before reads */
    //    for (uint32_t i = 0; i < config.num_txns; ++i)
    //    {
    //        BaseTxn *txn = txn_array.getTxn(index_epoch_id, i);
    //        BaseTxn *txn_param = index_output.getTxn(i);
    //        index->indexTxnWrites(txn, txn_param, index_epoch_id);
    //    }
    //    for (uint32_t i = 0; i < config.num_txns; ++i)
    //    {
    //        BaseTxn *txn = txn_array.getTxn(index_epoch_id, i);
    //        BaseTxn *txn_param = index_output.getTxn(i);
    //        index->indexTxnReads(txn, txn_param, index_epoch_id);
    //    }
}

// GPU-stager construction (hybrid allocator + 8 per-table stagers +
// dirty/delivered wiring), factored out of the ctor. Requires GPU
// records/versions and cpu_records to exist.
void TpccDb::initGpuStagersAndWireDirty()
{
    // Allocator member kept alive for the life of the stagers; it is only
    // used during constructor allocations today, but per-epoch allocations
    // could route through the same Allocator&.
    hybrid_allocator = std::make_unique<GpuAllocator>();

    warehouse_stager  = std::make_shared<WarehouseStager>(*hybrid_allocator,
        records.warehouse_record,  versions.warehouse_version,  cpu_records.warehouse_record,
        static_cast<uint32_t>(config.cacheCapacityWarehouse()),  static_cast<uint32_t>(config.warehouseTableSize()));
    district_stager   = std::make_shared<DistrictStager>(*hybrid_allocator,
        records.district_record,   versions.district_version,   cpu_records.district_record,
        static_cast<uint32_t>(config.cacheCapacityDistrict()),   static_cast<uint32_t>(config.districtTableSize()));
    customer_stager   = std::make_shared<CustomerStager>(*hybrid_allocator,
        records.customer_record,   versions.customer_version,   cpu_records.customer_record,
        static_cast<uint32_t>(config.cacheCapacityCustomer()),   static_cast<uint32_t>(config.customerTableSize()));
    item_stager       = std::make_shared<ItemStager>(*hybrid_allocator,
        records.item_record,       versions.item_version,       cpu_records.item_record,
        static_cast<uint32_t>(config.cacheCapacityItem()),       static_cast<uint32_t>(config.itemTableSize()));
    stock_stager      = std::make_shared<StockStager>(*hybrid_allocator,
        records.stock_record,      versions.stock_version,      cpu_records.stock_record,
        static_cast<uint32_t>(config.cacheCapacityStock()),      static_cast<uint32_t>(config.stockTableSize()));
    new_order_stager  = std::make_shared<NewOrderStager>(*hybrid_allocator,
        records.new_order_record,  versions.new_order_version,  cpu_records.new_order_record,
        static_cast<uint32_t>(config.cacheCapacityNewOrder()),   static_cast<uint32_t>(config.newOrderTableSize()),
        /*enable_reclaim_eviction=*/config.txn_mix.delivery > 0);
    order_stager      = std::make_shared<OrderStager>(*hybrid_allocator,
        records.order_record,      versions.order_version,      cpu_records.order_record,
        static_cast<uint32_t>(config.cacheCapacityOrder()),      static_cast<uint32_t>(config.orderTableSize()));
    order_line_stager = std::make_shared<OrderLineStager>(*hybrid_allocator,
        records.order_line_record, versions.order_line_version, cpu_records.order_line_record,
        static_cast<uint32_t>(config.cacheCapacityOrderLine()),  static_cast<uint32_t>(config.orderLineTableSize()),
        /*enable_reclaim_eviction=*/true);
    // Plumb the OL reclaim-flag pointer into TpccRecords so the hybrid
    // Delivery executor can mark slots delivered (= reclaimable) after
    // writing OL_DELIVERY_D.
    records.order_line_delivered_flag = order_line_stager->reclaim_flag_ptr();

    // Plumb each stager's per-slot dirty arrays into TpccRecords so the
    // executor's gpuWriteToTableCoop / gpuWriteToTableThread can mark
    // dirty inline. Replaces the staging-time mark_dirty pipeline that
    // ran four GPU kernels per stager per epoch.
    records.warehouse_dirty_v1  = warehouse_stager ->dirty_v1_ptr(); records.warehouse_dirty_v2  = warehouse_stager ->dirty_v2_ptr();
    records.district_dirty_v1   = district_stager  ->dirty_v1_ptr(); records.district_dirty_v2   = district_stager  ->dirty_v2_ptr();
    records.customer_dirty_v1   = customer_stager  ->dirty_v1_ptr(); records.customer_dirty_v2   = customer_stager  ->dirty_v2_ptr();
    records.new_order_dirty_v1  = new_order_stager ->dirty_v1_ptr(); records.new_order_dirty_v2  = new_order_stager ->dirty_v2_ptr();
    records.order_dirty_v1      = order_stager     ->dirty_v1_ptr(); records.order_dirty_v2      = order_stager     ->dirty_v2_ptr();
    records.order_line_dirty_v1 = order_line_stager->dirty_v1_ptr(); records.order_line_dirty_v2 = order_line_stager->dirty_v2_ptr();
    records.item_dirty_v1       = item_stager      ->dirty_v1_ptr(); records.item_dirty_v2       = item_stager      ->dirty_v2_ptr();
    records.stock_dirty_v1      = stock_stager     ->dirty_v1_ptr(); records.stock_dirty_v2      = stock_stager     ->dirty_v2_ptr();

    hybrid_allocator->PrintMemoryInfo();
}

// GPU records/versions allocation (per-table cache pools + version
// scratchpads), factored out of the ctor.
void TpccDb::allocGpuRecordsAndVersions()
{
    GpuAllocator allocator;
    auto &logger = Logger::GetInstance();
    /* CAUTION: version size is based on the number of transactions, and will cause sync issue if too small */
    size_t warehouse_rec_size = sizeof(Record<WarehouseValue>) * config.cacheCapacityWarehouse();
    size_t warehouse_ver_size = sizeof(Version<WarehouseValue>) * config.num_txns;
    logger.Info("Warehouse record: {}, version: {}", formatSizeBytes(warehouse_rec_size),
        formatSizeBytes(warehouse_ver_size));
    records.warehouse_record = static_cast<Record<WarehouseValue> *>(allocator.Allocate(warehouse_rec_size));
    versions.warehouse_version = static_cast<Version<WarehouseValue> *>(allocator.Allocate(warehouse_ver_size));

    size_t district_rec_size = sizeof(Record<DistrictValue>) * config.cacheCapacityDistrict();
    size_t district_ver_size = sizeof(Version<DistrictValue>) * config.num_txns;
    logger.Info(
        "District record: {}, version: {}", formatSizeBytes(district_rec_size), formatSizeBytes(district_ver_size));
    records.district_record = static_cast<Record<DistrictValue> *>(allocator.Allocate(district_rec_size));
    versions.district_version = static_cast<Version<DistrictValue> *>(allocator.Allocate(district_ver_size));

    size_t customer_rec_size = sizeof(Record<CustomerValue>) * config.cacheCapacityCustomer();
    size_t customer_ver_size = sizeof(Version<CustomerValue>) * config.num_txns;
    logger.Info(
        "Customer record: {}, version: {}", formatSizeBytes(customer_rec_size), formatSizeBytes(customer_ver_size));
    records.customer_record = static_cast<Record<CustomerValue> *>(allocator.Allocate(customer_rec_size));
    versions.customer_version = static_cast<Version<CustomerValue> *>(allocator.Allocate(customer_ver_size));

    /* TODO: history table is too big */
    //        size_t history_rec_size = sizeof(Record<HistoryValue>) * config.historyTableSize();
    //        size_t history_ver_size = sizeof(Version<HistoryValue>) * config.historyTableSize();
    //        logger.Info("History record: {}, version: {}", formatSizeBytes(history_rec_size),
    //                    formatSizeBytes(history_ver_size));
    //        records.history_record = static_cast<Record<HistoryValue> *>(allocator.Allocate(history_rec_size));
    //        versions.history_version = static_cast<Version<HistoryValue> *>(allocator.Allocate(history_ver_size));

    size_t new_order_rec_size = sizeof(Record<NewOrderValue>) * config.cacheCapacityNewOrder();
    size_t new_order_ver_size = sizeof(Version<NewOrderValue>) * config.num_txns; /* TODO: not needed */
    logger.Info("NewOrder record: {}, version: {}", formatSizeBytes(new_order_rec_size),
        formatSizeBytes(new_order_ver_size));
    records.new_order_record = static_cast<Record<NewOrderValue> *>(allocator.Allocate(new_order_rec_size));
    versions.new_order_version = static_cast<Version<NewOrderValue> *>(allocator.Allocate(new_order_ver_size));

    size_t order_rec_size = sizeof(Record<OrderValue>) * config.cacheCapacityOrder();
    size_t order_ver_size = sizeof(Version<OrderValue>) * config.num_txns; /* TODO: not needed */
    logger.Info("Order record: {}, version: {}", formatSizeBytes(order_rec_size), formatSizeBytes(order_ver_size));
    records.order_record = static_cast<Record<OrderValue> *>(allocator.Allocate(order_rec_size));
    versions.order_version = static_cast<Version<OrderValue> *>(allocator.Allocate(order_ver_size));

    size_t order_line_rec_size = sizeof(Record<OrderLineValue>) * config.cacheCapacityOrderLine();
    size_t order_line_ver_size = sizeof(Version<OrderLineValue>) * config.num_txns * 15; /* TODO: not needed */
    logger.Info("OrderLine record: {}, version: {}", formatSizeBytes(order_line_rec_size),
        formatSizeBytes(order_line_ver_size));
    records.order_line_record = static_cast<Record<OrderLineValue> *>(allocator.Allocate(order_line_rec_size));
    versions.order_line_version = static_cast<Version<OrderLineValue> *>(allocator.Allocate(order_line_ver_size));

    size_t item_rec_size = sizeof(Record<ItemValue>) * config.cacheCapacityItem();
    size_t item_ver_size = sizeof(Version<ItemValue>) * config.num_txns * 15; /* TODO: not needed */
    logger.Info("Item record: {}, version: {}", formatSizeBytes(item_rec_size), formatSizeBytes(item_ver_size));
    records.item_record = static_cast<Record<ItemValue> *>(allocator.Allocate(item_rec_size));
    versions.item_version = static_cast<Version<ItemValue> *>(allocator.Allocate(item_ver_size));

    size_t stock_rec_size = sizeof(Record<StockValue>) * config.cacheCapacityStock();
    size_t stock_ver_size = sizeof(Version<StockValue>) * config.num_txns * 15;
    logger.Info("Stock record: {}, version: {}", formatSizeBytes(stock_rec_size), formatSizeBytes(stock_ver_size));
    records.stock_record = static_cast<Record<StockValue> *>(allocator.Allocate(stock_rec_size));
    versions.stock_version = static_cast<Version<StockValue> *>(allocator.Allocate(stock_ver_size));

    allocator.PrintMemoryInfo();
}

// GPU planners + submitter construction, factored out of the ctor.
void TpccDb::initGpuPlannersAndSubmitter()
{
    GpuAllocator allocator;
    warehouse_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
        "warehouse", allocator, 0, 2, config.num_txns, config.num_warehouses, initialization_output);
    district_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
        "district", allocator, 0, 2, config.num_txns, config.num_warehouses * 10, initialization_output);
    customer_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
        "customer", allocator, 0, 20, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
    history_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
        "history", allocator, 0, 1, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
    new_order_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
        "new_order", allocator, 0, 10, config.num_txns, config.num_warehouses * 10 * 900, initialization_output);
    order_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
        "order", allocator, 0, 20, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
    order_line_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>("order_line", allocator,
        0, 30, config.num_txns, config.num_warehouses * 10 * 3000 * 15, initialization_output);
    item_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
        "item", allocator, 0, 15, config.num_txns, 100'000, initialization_output);
    stock_planner = std::make_unique<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>>(
        "stock", allocator, 0, 15 * 2, config.num_txns, 100'000 * config.num_warehouses, initialization_output);

    warehouse_planner->Initialize();
    district_planner->Initialize();
    customer_planner->Initialize();
    history_planner->Initialize();
    new_order_planner->Initialize();
    order_planner->Initialize();
    order_line_planner->Initialize();
    item_planner->Initialize();
    stock_planner->Initialize();
    allocator.PrintMemoryInfo();

    using SubmitDestT = TpccGpuSubmitter<TpccTxnParamArrayT>::TableSubmitDest;
    // Helper to keep the 9 SubmitDestT initializers from drowning the
    // call site. Each per-table planner exposes the same set of
    // working-set arrays (added during the YCSB prototype); we wire them
    // through unchanged here.
    auto mk_dest = [](auto& p) {
        // The 9 planners are TableExecutionPlanner&; max_num_ops is a
        // GPU-side field on the derived GpuTableExecutionPlanner, so we
        // downcast (this branch only runs when initialize_device == GPU).
        auto* gp = static_cast<GpuTableExecutionPlanner<TpccTxnExecPlanArrayT>*>(p.get());
        return SubmitDestT{
            p->d_num_ops, p->d_op_offsets, p->d_submitted_ops,
            p->d_scratch_array, p->scratch_array_bytes,
            p->curr_num_ops, p->curr_num_read_ops, p->curr_num_write_ops, p->curr_num_insert_ops,
            p->d_all_keys, p->d_read_keys, p->d_write_keys, p->d_insert_keys, p->d_op_is_insert,
            gp->max_num_ops
        };
    };
    submitter = std::make_shared<TpccGpuSubmitter<TpccTxnParamArrayT>>(
        mk_dest(warehouse_planner), mk_dest(district_planner),
        mk_dest(customer_planner),  mk_dest(history_planner),
        mk_dest(new_order_planner), mk_dest(order_planner),
        mk_dest(order_line_planner), mk_dest(item_planner),
        mk_dest(stock_planner));
}
} // namespace epic::tpcc
