//
// Created by Shujian Qian on 2023-08-23.
//

#include <algorithm>
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

namespace {
    template <typename VariantPtr>
    void LogVariantPtr(const char* name, const VariantPtr& v) {
        auto& logger = epic::Logger::GetInstance();
        std::visit([&](auto* p) {
            logger.Info("{} address: {}", name, static_cast<const void*>(p));
        }, v);
    }
} // namespace

namespace epic::ycsb {

YcsbBenchmark::YcsbBenchmark(YcsbConfig config)
    : config(config)
    , txn_array(config.epochs)
    , index_input(config.num_txns, config.index_device, false)
    , index_output(config.num_txns, config.index_device)
    , initialization_input(config.num_txns, config.initialize_device, false)
    , initialization_output(config.num_txns, config.initialize_device)
    , execution_param_input(config.num_txns, config.execution_device, false)
    , execution_plan_input(config.num_txns, config.execution_device, false)
{
    alloc_ = std::make_shared<GpuAllocator>();
    auto &logger = Logger::GetInstance();

    // Delete-bearing mixes are supported on the hybrid staging path with
    // whole-record storage only. The other executors' op switches have no
    // delete arm, and the split-field per-field expansion has no delete
    // encoding; fail at construction rather than at a kernel assert.
    if (config.txn_mix.num_deletes > 0
        && (config.execution_mode != ExecMode::HYBRID_STAGING || config.split_field))
    {
        throw std::runtime_error(
            "delete-bearing YCSB mixes require -y hybrid_staging with -f false (no field split)");
    }

    // Hybrid staging with async flush overlap is sensitive to CUDA host-side
    // sync latency: the default Auto mode parks the thread on a driver IRQ,
    // and wake-up from deep C-states on our CPU adds ~1 ms to every
    // cudaStreamSynchronize / cudaEventSynchronize call. On short-work paths
    // (idx_transfer at 120 B split, flush_sync_prev at 120 B non-split) this
    // tax is a large fraction of the sync wall-clock. Yield mode busy-loops
    // with sched_yield() and returns within microseconds of GPU completion.
    if (config.overlap_flush && config.execution_mode == ExecMode::HYBRID_STAGING)
    {
        setDeviceScheduleYield();

    }

    for (int i = 0; i < config.epochs; ++i)
    {
        txn_array[i] = TxnArray<YcsbTxn>(config.num_txns, DeviceType::CPU);
    }
    // Construct the CPU shadow first so the GPU index can bind a
    // reference to it. The shadow owns the durable host-side copy of
    // the key->CRID mapping, with a lifetime independent of the GPU
    // index.
    cpu_shadow_ = std::make_unique<YcsbCpuShadowIndex>(config);
    index = std::make_shared<YcsbGpuIndex>(config, *cpu_shadow_);
    input_index_bridge.Link(txn_array[0], index_input);
    index_initialization_bridge.Link(index_output, initialization_input);
    index_execution_param_bridge.Link(index_output, execution_param_input);
    initialization_execution_plan_bridge.Link(initialization_output, execution_plan_input);


    initGpuPlannerAndSubmitter();

    if (config.execution_mode == ExecMode::GPU_ONLY)
    {
        if (config.split_field)
        {
            size_t record_size = sizeof(YcsbFieldRecords) * config.num_records * 10;
            GPU_records = static_cast<YcsbFieldRecords *>(alloc_->Allocate(record_size));
            // print memory address of GPU_records
            LogVariantPtr("GPU_records", GPU_records);
            logger.Info("Field-split record size: {}", formatSizeBytes(record_size));
            size_t version_size = sizeof(YcsbFieldVersions) * config.num_records * 10;
            GPU_versions = static_cast<YcsbFieldVersions *>(alloc_->Allocate(version_size));
            logger.Info("Field-split version size: {}", formatSizeBytes(version_size));
        }
        else
        {
            size_t record_size = sizeof(YcsbRecords) * config.num_records;
            GPU_records = static_cast<YcsbRecords *>(alloc_->Allocate(record_size));
            logger.Info("Record size: {}", formatSizeBytes(record_size));
            size_t version_size = sizeof(YcsbVersions) * config.num_records;
            GPU_versions = static_cast<YcsbVersions *>(alloc_->Allocate(version_size));
            logger.Info("Version size: {}", formatSizeBytes(version_size));
        }
        alloc_->PrintMemoryInfo();

        executor =
            std::make_shared<GpuExecutor>(GPU_records, GPU_versions, execution_param_input, execution_plan_input, config);
        //        executor =
        //            std::make_shared<GpuExecutor>(records, versions, initialization_input, initialization_output,
        //            config);
    }
    else if (config.execution_mode == ExecMode::CPU_ONLY)
    {
        if (config.split_field)
        {
            // split_field on CPU: each record is broken into 10 independent
            // field-records (Record<YcsbFieldValue>), indexed as rec*10 + field.
            // sizeof(YcsbFieldRecords)  = 256 B (aligned to kDeviceCacheLineSize=64)
            // sizeof(YcsbFieldVersions) = 128 B (aligned to kDeviceCacheLineSize=64)
            const size_t num_units = static_cast<size_t>(config.num_records) * 10;
            size_t record_size = sizeof(YcsbFieldRecords) * num_units;
            CPU_records = static_cast<YcsbFieldRecords *>(Malloc(record_size));
            logger.Info("Split-field record size: {}", formatSizeBytes(record_size));
            size_t version_size = sizeof(YcsbFieldVersions) * num_units;
            CPU_versions = static_cast<YcsbFieldVersions *>(Malloc(version_size));
            logger.Info("Split-field version size: {}", formatSizeBytes(version_size));
        }
        else
        {
            size_t record_size = sizeof(YcsbRecords) * config.num_records;
            CPU_records = static_cast<YcsbRecords *>(Malloc(record_size));
            logger.Info("Record size: {}", formatSizeBytes(record_size));
            size_t version_size = sizeof(YcsbVersions) * config.num_records;
            CPU_versions = static_cast<YcsbVersions *>(Malloc(version_size));
            logger.Info("Version size: {}", formatSizeBytes(version_size));
        }

        executor =
            std::make_shared<CpuExecutor>(CPU_records, CPU_versions, execution_param_input, execution_plan_input, config);
    }
    else if (config.execution_mode == ExecMode::HYBRID_STAGING)
    {

        logger.Info("We are in the Hybrid-staging execution mode.");
        const uint32_t num_fields_per_record = config.split_field ? 10 : 1;
        const uint64_t num_units = static_cast<uint64_t>(config.num_records) * num_fields_per_record;

        const size_t rec_size = config.split_field ? sizeof(YcsbFieldRecords) : sizeof(YcsbRecords);
        const size_t ver_size = config.split_field ? sizeof(YcsbFieldVersions) : sizeof(YcsbVersions);

        const size_t per_unit_payload = rec_size;

        size_t free, total;
        alloc_->GetMemoryInfo(free, total);
        // HBM headroom held back from the cache budget to avoid OOM
        constexpr size_t kSafetyMargin = 8ULL * 1024 * 1024 * 1024;
        size_t usable = free > kSafetyMargin ? free - kSafetyMargin : 0;
        // We also allocate ver_size * config.num_txns * 10 on the GPU for versions, so we must subtract that too
        size_t ver_allocation = ver_size * config.num_txns * 10;
        usable = usable > ver_allocation ? usable - ver_allocation : 0;

        // Subtract the fixed-cost CRID->GRID flat index array (not per-slot).
        const size_t flat_index_bytes = num_units * sizeof(uint32_t);
        usable = usable > flat_index_bytes ? usable - flat_index_bytes : 0;

        // Let the stager compute capacity — it knows its own per-slot overhead.
        uint64_t staging_capacity = HybridStager::compute_staging_capacity(
            usable, per_unit_payload, static_cast<uint32_t>(num_units));

        // EPIC_YCSB_CACHE_CAP overrides the auto-sized capacity DOWNWARD,
        // pinning the cache size for cache-ratio sweeps and the
        // matched-cache-pressure figure convention. Env unset = the
        // autosized capacity above.
        if (const char* env_cap = std::getenv("EPIC_YCSB_CACHE_CAP")) {
            uint64_t override_cap = std::strtoull(env_cap, nullptr, 10);
            if (override_cap > 0 && override_cap < staging_capacity) {
                logger.Info("EPIC_YCSB_CACHE_CAP override: {} -> {}", staging_capacity, override_cap);
                staging_capacity = override_cap;
            }
        }

        logger.Info("GPU memory: Free-->{}, Total-->{}, usable-->{}, staging capacity-->{} {}",
        formatSizeBytes(free), formatSizeBytes(total), formatSizeBytes(usable),
        staging_capacity, config.split_field ? "field-units" : "records");

        // --- CPU primary store (plain host memory, allocated once here) ---
        // Route through MallocDurable: gated by EPIC_DURABLE_STORE/
        // EPIC_RECOVER_FROM, byte-identical to Malloc when neither is set.
        void* CPU_recs_raw = MallocDurable("ycsb_records", rec_size * num_units);
        logger.Info("CPU payload sizes: recs={}", formatSizeBytes(rec_size * num_units));
        alloc_->PrintMemoryInfoCpu();
        CPU_records = config.split_field
            ? YcsbRecordArrType{ static_cast<ycsb::YcsbFieldRecords*>(CPU_recs_raw) }
            : YcsbRecordArrType{ static_cast<ycsb::YcsbRecords*>(CPU_recs_raw) };
        // Persist the autosized cache capacity to the member config (the
        // ctor parameter shadows it; see the TpccDb analog of this pattern).
        this->config.gpu_capacity = static_cast<uint32_t>(staging_capacity);
        // GPU cache + stager + executor.
        initGpuHybridStagerExecutor();

    }
    else
    {
        throw std::runtime_error("epic::ycsb::YcsbBenchmark::YcsbBenchmark() found unknown execution device.");
    }
}


void YcsbBenchmark::loadInitialData()
{
    // Load the CPU shadow first so the GPU index's post-load smoke
    // check sees the shadow populated.
    if (cpu_shadow_) cpu_shadow_->loadInitialData();
    index->loadInitialData();

}

void YcsbBenchmark::generateTxns()
{
    auto &logger = Logger::GetInstance();
    std::random_device rd;
#ifdef EGAD_VALIDATION
    // Optional deterministic seeding via EPIC_YCSB_SEED for replay-correctness
    // paired runs (no-crash vs crash+replay must see identical txn_array[]).
    // When unset, fall back to entropy. Validation build only.
    const char* seed_env = std::getenv("EPIC_YCSB_SEED");
    const uint64_t base_seed = seed_env ? std::strtoull(seed_env, nullptr, 10)
                                        : static_cast<uint64_t>(rd());
    if (seed_env) {
        logger.Info("EPIC_YCSB_SEED={} (deterministic txn generation)", base_seed);
    }
#else
    const uint64_t base_seed = static_cast<uint64_t>(rd());
#endif // EGAD_VALIDATION
    std::mt19937 gen(static_cast<uint32_t>(base_seed));
    std::uniform_int_distribution<> percentage_gen(0, 99);
    std::uniform_int_distribution<> field_gen(0, 9);
    ZipfianRandom zipf;
    // Derive zipf seed from base via a splitmix-style mix so the two
    // PRNGs don't share state but are both deterministic given base_seed.
    const uint64_t zipf_seed = base_seed ^ 0x9e3779b97f4a7c15ULL;
    zipf.init(config.num_records, config.skew_factor, static_cast<uint32_t>(zipf_seed));
    logger.Info("GENERATE TXNS - Num records: {}", config.num_records);
    uint32_t max_existing_record = static_cast<uint32_t>(config.starting_num_records);
    // print out the full record read config

    auto getOpType = [&](int percentage) -> YcsbOpType {
        int acc = config.txn_mix.num_reads;
        if (percentage < acc)
        {
            return config.full_record_read ? YcsbOpType::FULL_READ : YcsbOpType::READ;
        }
        acc += config.txn_mix.num_writes;
        if (percentage < acc)
        {
            return YcsbOpType::UPDATE;
        }
        acc += config.txn_mix.num_rmw;
        if (percentage < acc)
        {
            return config.full_record_read ? YcsbOpType::FULL_READ_MODIFY_WRITE : YcsbOpType::READ_MODIFY_WRITE;
        }
        acc += config.txn_mix.num_deletes;
        if (percentage < acc)
        {
            return YcsbOpType::DELETE;
        }
        return YcsbOpType::INSERT;
    };

    const int recent_bias_threshold =
        static_cast<int>(std::clamp(config.recent_read_bias, 0.0, 1.0) * 100.0);
    const uint32_t recent_window = config.recent_window_size;
    std::uniform_int_distribution<uint32_t> recent_offset_gen(
        0, recent_window > 0 ? recent_window - 1 : 0);

    // Sliding-window generation for delete-bearing mixes (ycsbw). A
    // separate branch so mixes without deletes keep the loop below
    // bit-identical (its RNG stream backs the recovery anchors). Inserts
    // mint keys at the head of the key space; each delete consumes the
    // oldest live key from the tail; reads and updates draw from the live
    // window [tail, head) only. Two passes per transaction: pass 1 fixes
    // op types and the head/tail keys, so pass 2's window draws can never
    // land on a key this transaction inserts or deletes, which keeps
    // every delete terminal (no operation ordered after a delete ever
    // touches its key, in this epoch or any later one).
    if (config.txn_mix.num_deletes > 0)
    {
        uint32_t delete_tail = 0;
        for (size_t epoch = 0; epoch < config.epochs; ++epoch)
        {
            logger.Info("Generating epoch {}", epoch);
            for (size_t t = 0; t < config.num_txns; ++t)
            {
                BaseTxn *base_txn = txn_array[epoch].getTxn(t);
                YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(base_txn->data);
                for (size_t op = 0; op < config.num_ops_per_txn; ++op)
                {
                    txn->ops[op] = getOpType(percentage_gen(gen));
                    if (txn->ops[op] == YcsbOpType::INSERT)
                    {
                        txn->keys[op] = max_existing_record++;
                    }
                    else if (txn->ops[op] == YcsbOpType::DELETE)
                    {
                        if (delete_tail >= max_existing_record)
                        {
                            throw std::runtime_error(
                                "ycsb delete generation drained the live window; "
                                "the delete rate must not exceed insert rate + initial population");
                        }
                        txn->keys[op] = delete_tail++;
                    }
                    else
                    {
                        txn->keys[op] = 0xffffffffu;  // drawn in pass 2
                    }
                }
                for (size_t op = 0; op < config.num_ops_per_txn; ++op)
                {
                    if (txn->ops[op] == YcsbOpType::INSERT || txn->ops[op] == YcsbOpType::DELETE)
                    {
                        continue;
                    }
                    bool retry;
                    do
                    {
                        const uint32_t window = max_existing_record - delete_tail;
                        const uint32_t rank = static_cast<uint32_t>(zipf.next()) % window;
                        txn->keys[op] = max_existing_record - 1u - rank;  // hot end = newest live keys
                        retry = false;
                        for (size_t j = 0; j < config.num_ops_per_txn; ++j)
                        {
                            if (j != op && txn->keys[j] == txn->keys[op])
                            {
                                retry = true;
                                break;
                            }
                        }
                    } while (retry);
                    txn->fields[op] = field_gen(gen);
                }
            }
        }
        return;
    }

    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        logger.Info("Generating epoch {}", epoch);
        for (size_t i = 0; i < config.num_txns; ++i)
        {
            BaseTxn *base_txn = txn_array[epoch].getTxn(i);
            uint32_t timestamp = epoch * config.num_txns + i;
            YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(base_txn->data);
            for (int i = 0; i < config.num_ops_per_txn; ++i)
            {
                int percentage = percentage_gen(gen);
                txn->ops[i] = getOpType(percentage);
                if (txn->ops[i] == YcsbOpType::INSERT)
                {
                    txn->keys[i] = max_existing_record;
                    ++max_existing_record;
                    continue;
                }

                bool retry;
                do
                {
                    const bool use_recent_bias =
                        recent_bias_threshold > 0
                        && recent_window > 0
                        && max_existing_record > recent_window
                        && percentage_gen(gen) < recent_bias_threshold;
                    if (use_recent_bias)
                    {
                        // Uniform draw from the last recent_window inserted CRIDs
                        // (max_existing_record - 1 is the most recent allocation).
                        txn->keys[i] = max_existing_record - 1 - recent_offset_gen(gen);
                    }
                    else
                    {
                        do
                        {
                            txn->keys[i] = zipf.next();
                        } while (txn->keys[i] >= max_existing_record);
                    }
                    retry = false;
                    for (int j = 0; j < i; ++j)
                    {
                        if (txn->keys[i] == txn->keys[j])
                        {
                            retry = true;
                            break;
                        }
                    }
                } while (retry);
                txn->fields[i] = field_gen(gen);
            }
        }
    }
}

// Per-epoch pipeline, shared by the normal run loop and the
// recovery replay path.

// Within-epoch crash injector. Fires a REAL GPU fault at a
// chosen (EPIC_CRASH_AT_EPOCH, EPIC_CRASH_AT_PHASE) so the recovery sweep can
// crash at different points in the pipeline: phase 0 = top of epoch (before
// runEpoch); 1 = after admission/staging; 2 = after execution; 3 = after E's
// writeback is queued (marker E+1, bytes landing); 4 = E-1 fully drained,
// before the marker bump (marker E, none of E durable); 5 = after the bump,
// before E's writeback starts (marker E+1, none of E durable). Covering
// distinct durable states models the untimed-ness of a real fault. Never fires
// in recover-mode. EPIC_CRASH_AT_PHASE defaults to 0.
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

void YcsbBenchmark::runEpoch(uint32_t epoch_id, FlushHandle& flush_inflight)
{
    auto &logger = Logger::GetInstance();
    std::chrono::high_resolution_clock::time_point start_time, end_time;

    /* transfer */
    {
        start_time = std::chrono::high_resolution_clock::now();
        uint32_t index_epoch_id = epoch_id - 1;
        input_index_bridge.Link(txn_array[index_epoch_id], index_input);
        input_index_bridge.StartTransfer();
        input_index_bridge.FinishTransfer();
        end_time = std::chrono::high_resolution_clock::now();
        logger.Info("Epoch {} index_transfer time: {} us", epoch_id,
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
        index_execution_param_bridge.StartTransfer();
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
    // log that we are starting the staging phase
    logger.Info("Epoch {} starting staging phase", epoch_id);

    /* staging */
    {
        // only do staging in hybrid mode
        if (config.execution_mode == ExecMode::HYBRID_STAGING)
        {
            //start out by sorting d_all_keys(which is the union of all keys needed for this epoch)

            const uint32_t n_all    = planner->curr_num_ops;
            const uint32_t n_read   = planner->curr_num_read_ops;
            const uint32_t n_write  = planner->curr_num_write_ops;
            const uint32_t n_insert = planner->curr_num_insert_ops;
            logger.Info("Epoch {} key counts: all={}, read={}, write={}, insert={}",
                        epoch_id, n_all, n_read, n_write, n_insert);

            start_time = std::chrono::high_resolution_clock::now();
            {
                // Flag this epoch's deleted records' cache slots
                // reclaim-first before eviction runs. Safe within the
                // deleting epoch itself: any slot the epoch still touches
                // is needed-protected, and a slot with an in-flight
                // writeback is pinned, so the flag only accelerates
                // draining slots nothing needs anymore.
                if (config.txn_mix.num_deletes > 0) {
                    if (auto* gi = dynamic_cast<YcsbGpuIndex*>(index.get())) {
                        stager->mark_reclaimable(gi->deleteCridsDevice(), gi->numDeletesThisEpoch());
                    }
                }
                // Pass flush handle so scatter can be spawned after SG transfer
                FlushHandle* fh = (config.overlap_flush && flush_inflight.valid) ? &flush_inflight : nullptr;
                stager->prepareEpoch(epoch_id,
                     execution_param_input,
                     execution_plan_input,
                     planner->d_all_keys,  n_all,
                     planner->d_read_keys, n_read,
                     planner->d_write_keys, n_write,
                     planner->d_insert_keys, n_insert,
                     fh);
            }
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} staging time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
    }

    // crash point 1: after admission/staging (records in cache,
    // E-1's writeback still in flight, E not yet written back).
    maybeCrashAt(epoch_id, 1);

    /* initialize */
    {
        start_time = std::chrono::high_resolution_clock::now();
        planner->InitializeExecutionPlan();
        planner->FinishInitialization();
        end_time = std::chrono::high_resolution_clock::now();
        logger.Info("Epoch {} initialization time: {} us", epoch_id,
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    }

    /* transfer */
    {
        start_time = std::chrono::high_resolution_clock::now();
        initialization_execution_plan_bridge.StartTransfer();
        index_execution_param_bridge.FinishTransfer();
        initialization_execution_plan_bridge.FinishTransfer();
        end_time = std::chrono::high_resolution_clock::now();
        logger.Info("Epoch {} exec_transfer time: {} us", epoch_id,
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    }

    /* execution */
    {
        start_time = std::chrono::high_resolution_clock::now();
        logger.Info("Executing epoch {}", epoch_id);

        executor->execute(epoch_id);
        logger.Info("Epoch {} execution finished", epoch_id);

        end_time = std::chrono::high_resolution_clock::now();
        logger.Info("Epoch {} execution time: {} us", epoch_id,
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    }

    // crash point 2: after execution (E's writes are in the GPU
    // cache, NOT yet written back; E-1's writeback still in flight).
    maybeCrashAt(epoch_id, 2);

    /* periodic flush */
    {
        if (config.execution_mode == ExecMode::HYBRID_STAGING)
        {
            if (config.overlap_flush)
            {
                // 1) Retire flush(E-1): join scatter worker, clear dirty bits
                if (flush_inflight.valid)
                {
                    start_time = std::chrono::high_resolution_clock::now();
                    stager->sync_flush(flush_inflight);
                    end_time = std::chrono::high_resolution_clock::now();
                    logger.Info("Epoch {} flush_sync_prev time: {} us", epoch_id,
                        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
                }

                // crash point 4: E-1 drained (end-of-(E-1) fully durable),
                // E's writeback not started, marker still E. Recover rolls
                // back to end-of-(E-2) and replays E-1, E.
                maybeCrashAt(epoch_id, 4);

                // Advance the crash marker to E+1 between the drain and the
                // start, the one point where both directions are safe:
                // sync_flush has retired E-1 (end-of-(E-1) fully durable)
                // and E's worker has not been spawned (no end-of-(E-2) slot
                // overwritten yet). Earlier, E-1 could still be landing;
                // later, E's scatter is already overwriting end-of-(E-2)
                // slots (the writeback lands in the older slot).
                bumpRecoveryMarker(epoch_id + 1);

                // crash point 5: marker E+1, E's writeback not started.
                // Recover rolls back to end-of-(E-1), replays E.
                maybeCrashAt(epoch_id, 5);

                // 2) Setup + GPU pack (sync) + queue D2H + spawn worker
                // (scatter runs during the next epoch). The worker's first
                // Primary Store byte lands strictly after the bump above.
                start_time = std::chrono::high_resolution_clock::now();
                stager->start_flush_epoch_async(epoch_id, flush_inflight);
                end_time = std::chrono::high_resolution_clock::now();
                logger.Info("Epoch {} flush_start_async time: {} us", epoch_id,
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
                // crash point 3: E's writeback queued + landing, marker E+1.
                maybeCrashAt(epoch_id, 3);
            } else
            {
                // crash point 4 (sync path): marker still E, E's writeback
                // not started. Recover rolls back to end-of-(E-2), replays
                // E-1, E.
                maybeCrashAt(epoch_id, 4);
                // Advance the marker BEFORE the writeback: E-1 is fully
                // durable (last epoch's periodicFlush completed
                // synchronously) and E has not started, so the whole flush
                // below runs under marker E+1 and a crash anywhere inside it
                // rolls back to end-of-(E-1) and replays E. Bumped after the
                // flush instead, the entire synchronous writeback would land
                // under a marker that still reads E.
                bumpRecoveryMarker(epoch_id + 1);
                // crash point 5 (sync path): marker E+1, E's writeback not
                // started.
                maybeCrashAt(epoch_id, 5);
                start_time = std::chrono::high_resolution_clock::now();
                stager->periodicFlush(epoch_id);
                end_time = std::chrono::high_resolution_clock::now();
                logger.Info("Epoch {} flush time: {} us", epoch_id,
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
                // crash point 3 (sync path): epoch E fully written back,
                // marker E+1.
                maybeCrashAt(epoch_id, 3);
            }

        }

    }
}

void YcsbBenchmark::runBenchmark()
{
    FlushHandle flush_inflight;
    if (config.overlap_flush && config.execution_mode == ExecMode::HYBRID_STAGING)
    {
        Logger::GetInstance().Info("Overlapping flush enabled: flushes from epoch E may still be in-flight during epoch E+1");
    }

    auto &logger = Logger::GetInstance();

    const auto run_start = std::chrono::high_resolution_clock::now();

    // Durable recovery metadata + crash/recover wiring. All of
    // this is gated on a durable store being present (EPIC_DURABLE_STORE /
    // EPIC_RECOVER_FROM), so non-durable runs stay byte-identical. Normal run:
    // advance the durable crash marker once per epoch inside runEpoch.
    // Recover-mode: read E, demote the Primary Store to end-of-(E-2), and resume
    // the loop at E-1 (E-1,E = replay; rest = normal continuation).
    const bool kDurable = std::getenv("EPIC_DURABLE_STORE") || std::getenv("EPIC_RECOVER_FROM");
    const bool kRecover = std::getenv("EPIC_RECOVER_FROM") != nullptr;
    // Allocate the durable marker member ONCE (mmap-backed; zeroed on create,
    // PRESERVED on recover). The crash_epoch marker is advanced inside runEpoch
    // after E-1's writeback drains and before E's is started (bumpRecoveryMarker),
    // so phases 0-2 of epoch E read E (roll back to end-of-(E-2)) and phase 3
    // reads E+1 (roll back to end-of-(E-1)). On non-durable runs recovery_meta_
    // stays null, so the bump is a no-op and the run is byte-identical.
    if (kDurable && !recovery_meta_) {
        recovery_meta_ = static_cast<RecoveryMeta*>(MallocDurable("ycsb_meta", sizeof(RecoveryMeta)));
    }

    uint32_t start_epoch = 1;
    if (kRecover) {
        uint32_t E = 0;
        const YcsbRecoveryCursors* cursors = nullptr;
        if (recovery_meta_) recovery_meta_->read(E, cursors);
        if (E < 2) {
            logger.Warn("[RECOVER] crash marker E={} < 2 (no end-of-(E-2) to roll back to); "
                        "falling back to full replay from epoch 1", E);
        } else {
            const uint32_t f_e2 = cursors->free_start_p2;  // insert cursor at end-of-(E-2)
            const uint32_t d_e2 = cursors->del_p2;         // delete cursor at end-of-(E-2)
            logger.Info("[RECOVER] crash marker E={}; roll back to end-of-({}) (f_(E-2)={}), resume at epoch {}",
                        E, static_cast<int>(E) - 2, f_e2, E - 1);
            // 1) Demote the Primary Store's dual-version tags to end-of-(E-2).
            rollbackPrimaryStoreVersions(E - 1);
            // 2) Rebuild the index at end-of-(E-2): reconstruct the shadow
            //    shards from the durable insert-keys array, apply the durable
            //    delete log through d_(E-2) (deletes from E-1 and E are beyond
            //    the cursor, so their keys come back live and replay re-erases
            //    them), then upload to the GPU cuco map (which also resyncs
            //    both cursors). YCSB-F has no inserts or deletes so all three
            //    are no-ops.
            if (cpu_shadow_) {
                cpu_shadow_->reconstructInsertsFromDurable(f_e2);
                cpu_shadow_->applyDeletesFromDurable(d_e2);
#ifdef EGAD_VALIDATION
                // Liveness gate, reconstruction side: the shards must equal
                // an independent replay of the logs to the same cursors. A
                // lost or extra delete application is invisible to the store
                // byte hashes (deletes write no record bytes); this check is
                // what catches it.
                if (config.txn_mix.num_deletes > 0) {
                    cpu_shadow_->verifyLiveAgainstLogs(f_e2, d_e2);
                }
#endif // EGAD_VALIDATION
            }
            if (auto* gi = dynamic_cast<YcsbGpuIndex*>(index.get())) gi->rebuildCucoFromShadow(f_e2, d_e2);
            start_epoch = E - 1;
        }
    }

    for (uint32_t epoch_id = start_epoch; epoch_id <= config.epochs; ++epoch_id)
    {
        logger.Info("Running epoch {}", epoch_id);

        // Crash point 0: top of epoch (before the pipeline runs;
        // E-1's writeback overlapped/in-flight, E not started). maybeCrashAt
        // injects a REAL GPU fault at (EPIC_CRASH_AT_EPOCH, EPIC_CRASH_AT_PHASE);
        // phases 1-3 are inside runEpoch. Original run only; never the recover.
        maybeCrashAt(epoch_id, 0);

        runEpoch(epoch_id, flush_inflight);
    }
    // Final flush after all epochs are done
    if (config.overlap_flush && config.execution_mode == ExecMode::HYBRID_STAGING)
    {
        if (flush_inflight.valid)
        {
            // Release the last worker's between-halves wait (the admit
            // transfer it would wait for never comes after the final epoch).
            stager->signal_sg_transfer_done();
            auto start_time = std::chrono::high_resolution_clock::now();
            stager->sync_flush(flush_inflight);
            auto end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Final flush_sync_prev time: {} us",
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
    }

    // print out cpu and gpu memory info at the end of the benchmark
    alloc_->PrintMemoryInfoCpu();
    alloc_->PrintMemoryInfo();

    const auto run_end = std::chrono::high_resolution_clock::now();
    const double run_seconds =
        std::chrono::duration<double>(run_end - run_start).count();
    const uint64_t total_txns =
        static_cast<uint64_t>(config.epochs) * config.num_txns;
    if (run_seconds > 0.0) {
        Logger::GetInstance().Info("[RUN] {} epochs x {} txns = {} txns in {:.3f} s -> {:.0f} txns/s",
                                   config.epochs, config.num_txns, total_txns,
                                   run_seconds, total_txns / run_seconds);
    }

    if (config.execution_mode == ExecMode::HYBRID_STAGING && stager) {
        const uint64_t h2d = stager->pcie_admit_h2d_bytes();
        const uint64_t d2h = stager->pcie_writeback_d2h_bytes();
        const uint64_t admitted = stager->cache_slots_admitted_total();
        const uint64_t evicted  = stager->cache_slots_evicted_total();
        const uint64_t ring_pops = admitted - evicted;
        const double mb = 1024.0 * 1024.0;
        Logger::GetInstance().Info("[PCIE] cumulative payload over {} epochs: H2D admit={:.1f} MB, D2H writeback={:.1f} MB",
                                   config.epochs, h2d / mb, d2h / mb);
        Logger::GetInstance().Info("[PCIE] per-epoch average: H2D admit={:.1f} MB/epoch, D2H writeback={:.1f} MB/epoch",
                                   (h2d / mb) / config.epochs, (d2h / mb) / config.epochs);
        const uint64_t miss_admit = stager->cache_slots_miss_admit_total();
        const uint64_t insert_alloc = stager->cache_slots_insert_allocate_total();
        Logger::GetInstance().Info("[SLOTS] cumulative over {} epochs: admitted={} (ring_pops={} evictions={}, miss_admit={} insert_allocate={})",
                                   config.epochs, admitted, ring_pops, evicted, miss_admit, insert_alloc);
        Logger::GetInstance().Info("[SLOTS] per-epoch average: admitted={}/epoch (ring_pops={} evictions={}, miss_admit={} insert_allocate={})",
                                   admitted / config.epochs,
                                   ring_pops / config.epochs,
                                   evicted / config.epochs,
                                   miss_admit / config.epochs,
                                   insert_alloc / config.epochs);
    }

#ifdef EGAD_VALIDATION
    // State hash for replay-correctness verification. A no-crash baseline and a
    // crash+replay run at the same workload print identical hashes. Validation
    // build only.
    corruptPrimaryStoreForNegControl();   // no-op unless neg-control env set
    Logger::GetInstance().Info("[STATE-HASH] cpu_records FNV-1a 64-bit = 0x{:016x}",
                               hashCpuRecords());
    Logger::GetInstance().Info("[STATE-HASH-VALONLY] cpu_records placement-invariant = 0x{:016x}",
                               hashCpuRecordsValOnly());
    // Liveness gate, end-of-run side: digest of the live key->CRID mapping
    // derived from the durable logs. Deletes write no record bytes, so the
    // two store hashes above cannot distinguish a live key from a deleted
    // one; this line can. Durable runs only (the logs are the source).
    if (config.txn_mix.num_deletes > 0 && cpu_shadow_
        && (std::getenv("EPIC_DURABLE_STORE") || std::getenv("EPIC_RECOVER_FROM"))) {
        Logger::GetInstance().Info("[STATE-HASH-LIVE] ycsb live-mapping = 0x{:016x}",
                                   cpu_shadow_->liveDigestFromLogs(currentInsertCount(), currentDeleteCount()));
    }
#endif // EGAD_VALIDATION
    verifyInsertedRecords();
}


// GPU planner + submitter construction, factored out of the ctor.
void YcsbBenchmark::initGpuPlannerAndSubmitter()
{
    planner = std::make_shared<GpuTableExecutionPlanner<TxnArray<YcsbExecPlan>>>(
        "planner", *alloc_, 0, 11 * 10, config.num_txns, config.num_records, initialization_output);
    planner->Initialize();

    submitter = std::make_shared<YcsbGpuSubmitter>(
        YcsbSubmitter::TableSubmitDest{planner->d_num_ops, planner->d_op_offsets, planner->d_submitted_ops,
            planner->d_scratch_array, planner->scratch_array_bytes,
            planner->curr_num_ops, planner->curr_num_read_ops, planner->curr_num_write_ops, planner->curr_num_insert_ops,
            planner->d_all_keys, planner->d_read_keys, planner->d_write_keys, planner->d_insert_keys, planner->d_op_is_insert },
        config);
}

// Shared GPU cache (records/versions) + stager + executor construction for
// hybrid-staging. Uses config.gpu_capacity (set by the ctor autosizer) and
// the existing CPU_records.
void YcsbBenchmark::initGpuHybridStagerExecutor()
{
    const size_t rec_size = config.split_field ? sizeof(ycsb::YcsbFieldRecords)
                                               : sizeof(ycsb::YcsbRecords);
    const size_t ver_size = config.split_field ? sizeof(ycsb::YcsbFieldVersions)
                                               : sizeof(ycsb::YcsbVersions);
    const uint64_t num_units = static_cast<uint64_t>(config.num_records) *
                               (config.split_field ? 10ull : 1ull);
    const size_t ver_allocation = ver_size * config.num_txns * 10;

    void* GPU_recs_raw = alloc_->Allocate(rec_size * config.gpu_capacity);
    void* GPU_vers_raw = alloc_->Allocate(ver_allocation);
    GPU_records = config.split_field
        ? YcsbRecordArrType{ static_cast<ycsb::YcsbFieldRecords*>(GPU_recs_raw) }
        : YcsbRecordArrType{ static_cast<ycsb::YcsbRecords*>(GPU_recs_raw) };
    GPU_versions = config.split_field
        ? YcsbVersionArrType{ static_cast<ycsb::YcsbFieldVersions*>(GPU_vers_raw) }
        : YcsbVersionArrType{ static_cast<ycsb::YcsbVersions*>(GPU_vers_raw) };

    stager = std::make_shared<HybridStager>(
        *alloc_,
        GPU_records, GPU_versions, CPU_records,
        config.gpu_capacity,
        static_cast<uint32_t>(num_units),
        config.split_field,
        /*enable_reclaim_eviction=*/config.txn_mix.num_deletes > 0);

    executor = std::make_shared<HybridExecutor>(
        GPU_records, GPU_versions,
        config.gpu_capacity,
        execution_param_input, execution_plan_input, config);

    std::static_pointer_cast<HybridExecutor>(executor)->set_dirty_arrays(
        stager->dirty_v1_ptr(), stager->dirty_v2_ptr());
}
} // namespace epic::ycsb