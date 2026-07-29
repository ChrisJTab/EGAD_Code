//
// Created by Shujian Qian on 2023-11-22.
//

#include <cmath>
#include <memory>
#include <cuda/std/atomic>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <benchmarks/ycsb_gpu_index.h>
#include <benchmarks/tpcc_gpu_index.h>
#include <benchmarks/tpcc_table.h>
#include <gpu_txn.cuh>
#include <util_log.h>
#include <util_gpu_error_check.cuh>

#include <cuco/static_map.cuh>

#include <cub/cub.cuh>

namespace epic::ycsb {

namespace {

using YcsbIndexType = cuco::static_map<uint32_t, uint32_t>;
using YcsbIndexDeviceView = YcsbIndexType::device_view;

struct IsNotSentinel {
    __device__ bool operator()(uint32_t x) const {
        return x != 0xffffffffu; // matches what prepareYcsbIndexKernel writes for non-insert
    }
};


void __global__ prepareYcsbIndexKernel(GpuTxnArray txns, uint32_t *insert, uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *base_txn_ptr = txns.getTxn(tid);
    YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(base_txn_ptr->data);
    int base = tid * 10; // each txn has 10 ops
    for (int i = 0; i < 10; ++i)
    {
        if (txn->ops[i] == YcsbOpType::INSERT)
        {
            uint32_t key = txn->keys[i];
            insert[base + i] = key;
        }
        else
        {
            insert[base + i] = static_cast<uint32_t>(-1);
        }
    }
}

// Emit the epoch's delete set. For each DELETE op, writes the key and its
// resolved CRID (from the params the lookup kernel just filled) into
// per-op-slot arrays and raises the slot's flag; DeviceSelect::Flagged
// compacts both arrays with the same flags, preserving op order, so the
// durable delete log's append order is deterministic.
void __global__ prepareYcsbDeleteKernel(GpuTxnArray txns, GpuTxnArray params,
    uint32_t *keys, uint32_t *crids, uint8_t *flags, uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(txns.getTxn(tid)->data);
    YcsbTxnParam *param = reinterpret_cast<YcsbTxnParam *>(params.getTxn(tid)->data);
    int base = tid * 10;
    for (int i = 0; i < 10; ++i)
    {
        const bool is_delete = txn->ops[i] == YcsbOpType::DELETE;
        flags[base + i] = is_delete ? 1 : 0;
        if (is_delete)
        {
            keys[base + i] = txn->keys[i];
            crids[base + i] = param->record_ids[i];
        }
    }
}

// Translate each op's key to its record id via the index, copying ops and
// field ids through to the executor-facing params.
void __global__ indexYcsbKernel(GpuTxnArray txn, GpuTxnArray index, YcsbIndexDeviceView index_view, uint32_t num_txns)
{

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *base_txn_ptr = txn.getTxn(tid);
    BaseTxn *base_index_ptr = index.getTxn(tid);
    YcsbTxn *txn_ptr = reinterpret_cast<YcsbTxn *>(base_txn_ptr->data);
    YcsbTxnParam *index_ptr = reinterpret_cast<YcsbTxnParam *>(base_index_ptr->data);

    for (int i = 0; i < 10; ++i)
    {
        auto record_found = index_view.find(txn_ptr->keys[i]);
        if (record_found != index_view.end())
        {
            index_ptr->record_ids[i] = record_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            printf("record not found\n");
            assert(false);
        }
        index_ptr->ops[i] = txn_ptr->ops[i];
        index_ptr->field_ids[i] = txn_ptr->fields[i];
    }
}

class YcsbGpuIndexImpl
{
public:
    static constexpr double load_factor = 0.8;
//    static constexpr cuco::empty_key<uint32_t> empty_key_sentinel{static_cast<uint32_t>(-1)};
//    static constexpr cuco::empty_value<uint32_t> empty_value_sentinel{static_cast<uint32_t>(-1)};
        static constexpr cuco::empty_key<uint32_t> empty_key_sentinel{0xffffffff};
        static constexpr cuco::empty_value<uint32_t> empty_value_sentinel{0xffffffff};
        // Erased-slot sentinel for the delete path. Passed at BOTH map
        // construction sites (ctor and the recovery rebuild); cuco's erase
        // throws at runtime on a map built without it. Distinct from the
        // empty sentinel and outside the key universe (keys < num_records).
        static constexpr cuco::erased_key<uint32_t> erased_key_sentinel{0xfffffffe};

    YcsbConfig ycsb_config;

    // === CPU shadow index reference ===
    //
    // The host-side shadow (12 sharded maps, rollback snapshots, the
    // pinned host buffer for D2H of insert keys) lives in YcsbBenchmark
    // as a YcsbCpuShadowIndex. This class holds a reference. The shadow
    // is the host-side ground truth: after a crash, recovery repopulates
    // a fresh GPU index from it. Declared here (next to ycsb_config,
    // before the cuco map) so the ctor's member-init list order matches
    // declaration order.
    YcsbCpuShadowIndex& shadow_;

    std::shared_ptr<YcsbIndexType> index;
    YcsbIndexDeviceView index_view;

    uint32_t *d_free_rows;
    thrust::device_ptr<uint32_t> dp_free_rows;
    uint32_t free_start = 0;

    uint32_t *d_inserts, *d_valid_inserts;
    thrust::device_ptr<uint32_t> dp_inserts, dp_valid_inserts;
    uint32_t *d_num_insert;      // device-accessible pointer (mapped)
    uint32_t *h_num_insert;      // host-accessible pointer (mapped, same physical memory)

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Delete-set scratch, allocated only for delete-bearing mixes
    // (nullptr otherwise; mixes without deletes never run the delete
    // block). Per-op-slot key/CRID arrays + flags, their Flagged-compacted
    // outputs, a mapped count, and the cumulative delete-log cursor.
    uint32_t *d_deletes = nullptr, *d_valid_deletes = nullptr;
    uint32_t *d_delete_crids = nullptr, *d_valid_delete_crids = nullptr;
    uint8_t *d_delete_flags = nullptr;
    thrust::device_ptr<uint32_t> dp_valid_deletes;
    uint32_t *d_num_delete = nullptr;   // mapped device pointer
    uint32_t *h_num_delete = nullptr;   // mapped host pointer (same memory)
    uint32_t delete_count = 0;          // cumulative; the durable delete-log cursor
    uint32_t num_deletes_this_epoch = 0;
    void *d_flagged_temp = nullptr;
    size_t flagged_temp_bytes = 0;

    explicit YcsbGpuIndexImpl(YcsbConfig ycsb_config, YcsbCpuShadowIndex& shadow)
        : ycsb_config(ycsb_config)
        , shadow_(shadow)
        , index(std::make_shared<YcsbIndexType>(static_cast<size_t>(std::ceil(ycsb_config.num_records / load_factor)),
              empty_key_sentinel, empty_value_sentinel, erased_key_sentinel))
        , index_view(index->get_device_view())
    {
        auto &logger = Logger::GetInstance();
        // Allocate GPU memory for free rows

        const uint32_t remaining = ycsb_config.num_records - ycsb_config.starting_num_records;
        gpu_err_check(cudaMalloc(&d_free_rows, sizeof(uint32_t) * remaining));
        dp_free_rows = thrust::device_pointer_cast(d_free_rows);
        // Insert-key scratch, sized to the worst case (every op an INSERT);
        // DeviceSelect::If compacts the real inserts into d_valid_inserts.
        gpu_err_check(cudaMalloc(&d_inserts, sizeof(uint32_t) * ycsb_config.num_txns * ycsb_config.num_ops_per_txn));
        gpu_err_check(
            cudaMalloc(&d_valid_inserts, sizeof(uint32_t) * ycsb_config.num_txns * ycsb_config.num_ops_per_txn));
        dp_inserts = thrust::device_pointer_cast(d_inserts);
        dp_valid_inserts = thrust::device_pointer_cast(d_valid_inserts);
        gpu_err_check(cudaHostAlloc(&h_num_insert, sizeof(uint32_t), cudaHostAllocMapped));
        gpu_err_check(cudaHostGetDevicePointer(&d_num_insert, h_num_insert, 0));
        // Allocate temp storage for cub::DeviceSelect::If to filter out the non-insert ops
        IsNotSentinel pred{};
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, dp_inserts, dp_valid_inserts, d_num_insert,
            ycsb_config.num_txns * ycsb_config.num_ops_per_txn, pred);

        logger.Trace("Allocating {} bytes for temp storage", formatSizeBytes(temp_storage_bytes));
        gpu_err_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // Delete-set scratch, delete-bearing mixes only.
        if (ycsb_config.txn_mix.num_deletes > 0) {
            const size_t n_slots =
                static_cast<size_t>(ycsb_config.num_txns) * ycsb_config.num_ops_per_txn;
            gpu_err_check(cudaMalloc(&d_deletes, sizeof(uint32_t) * n_slots));
            gpu_err_check(cudaMalloc(&d_valid_deletes, sizeof(uint32_t) * n_slots));
            gpu_err_check(cudaMalloc(&d_delete_crids, sizeof(uint32_t) * n_slots));
            gpu_err_check(cudaMalloc(&d_valid_delete_crids, sizeof(uint32_t) * n_slots));
            gpu_err_check(cudaMalloc(&d_delete_flags, sizeof(uint8_t) * n_slots));
            dp_valid_deletes = thrust::device_pointer_cast(d_valid_deletes);
            gpu_err_check(cudaHostAlloc(&h_num_delete, sizeof(uint32_t), cudaHostAllocMapped));
            *h_num_delete = 0;
            gpu_err_check(cudaHostGetDevicePointer(&d_num_delete, h_num_delete, 0));
            cub::DeviceSelect::Flagged(d_flagged_temp, flagged_temp_bytes,
                d_deletes, d_delete_flags, d_valid_deletes, d_num_delete, n_slots);
            gpu_err_check(cudaMalloc(&d_flagged_temp, flagged_temp_bytes));
        }

        // CPU shadow allocation now lives in YcsbCpuShadowIndex's ctor
        // (host-side, owned by YcsbBenchmark). This class just holds a
        // reference.

        logger.Info("Finished constructing YcsbGpuIndex");
        size_t free, total;
        gpu_err_check(cudaMemGetInfo(&free, &total));
        logger.Info("GPU memory usage: {} / {}", formatSizeBytes(total - free), formatSizeBytes(total));
    }

    void loadInitialData()
    {
        auto &logger = Logger::GetInstance();
        logger.Info("Loading initial data(ycsb)");
        // create d_keys = [0, 1, 2, ..., starting_num_records - 1]
        // create d_values = [0, 1, 2, ..., starting_num_records - 1]
        // insert (d_keys[i], d_values[i]) into the index
        thrust::device_vector<uint32_t> d_keys(ycsb_config.starting_num_records);
        thrust::device_vector<uint32_t> d_values(ycsb_config.starting_num_records);
        thrust::sequence(d_keys.begin(), d_keys.end(), 0);
        thrust::sequence(d_values.begin(), d_values.end(), 0);
        logger.Info("Made it past sequences");
        auto zipped_kv = thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));
        logger.Info("Inserting initial data into index");
        index->insert(zipped_kv, zipped_kv + ycsb_config.starting_num_records); // insert into the index
        logger.Info("Inserted {} initial records into index", ycsb_config.starting_num_records);
        // verify the index by finding all keys and comparing the values
        thrust::device_vector<uint32_t> found_values(ycsb_config.starting_num_records);
        index->find(d_keys.begin(), d_keys.end(), found_values.begin()); //
        if (thrust::equal(d_values.begin(), d_values.end(), found_values.begin())) {
            logger.Info("Initial data loaded successfully");
        } else {
            logger.Error("Initial data loaded incorrectly");
        }

        if (empty_key_sentinel != 0xffffffff) {
            logger.Error("empty_key_sentinel is not 0xffffffff");
        } else {
            logger.Info("empty_key_sentinel is 0xffffffff");
        }

        // Free rows hold the insert CRIDs [starting_num_records, num_records).
        const uint32_t remaining = ycsb_config.num_records - ycsb_config.starting_num_records;
        thrust::sequence(dp_free_rows, dp_free_rows + remaining, ycsb_config.starting_num_records);

        // CPU shadow's loadInitialData (the sharded fill + the smoke
        // check) is called from YcsbBenchmark::loadInitialData before
        // this method, so the shadow is already populated by the time
        // we get here.

        logger.Info("Finished loading initial data");
        size_t free, total;
        gpu_err_check(cudaMemGetInfo(&free, &total));
        logger.Info("GPU memory usage: {} / {}", formatSizeBytes(total - free), formatSizeBytes(total));
    }

    void indexTxns(TxnArray<YcsbTxn> &txn_array, TxnArray<YcsbTxnParam> &index_array, uint32_t epoch_id)
    {
        if (txn_array.device != DeviceType::GPU || index_array.device != DeviceType::GPU)
        {
            throw std::runtime_error("TpccGpuIndex only supports GPU transaction array");
        }
        auto &logger = Logger::GetInstance();

        // Shift the shadow's trailing snapshots at the START of the
        // epoch so free_start_prev2_ holds the rollback target f_{E-2}
        // throughout this entire epoch (indexTxns, execution, flush).
        // Captures the value that was durable at the end of epochs
        // E-1 and E-2 before any of this epoch's bulk_insert
        // increments the cursor.
        shadow_.shiftSnapshotsAtEpochStart(free_start);

        constexpr uint32_t block_size = 512;

        prepareYcsbIndexKernel<<<(ycsb_config.num_txns + block_size - 1) / block_size, block_size>>>(
            GpuTxnArray(txn_array), d_inserts, ycsb_config.num_txns);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));

        // This filters d_inserts into d_valid_inserts, removing all -1u(0xffffffff) entries
        // d_num_insert will contain the number of valid inserts found
        IsNotSentinel pred{};
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_inserts, d_valid_inserts, d_num_insert,
            ycsb_config.num_txns * ycsb_config.num_ops_per_txn, pred);

        gpu_err_check(cudaStreamSynchronize(0));
        uint32_t num_inserts = *h_num_insert;  // read directly from mapped memory, no D2H needed

        logger.Info("Found {} inserts", num_inserts);
        const uint32_t free_total = ycsb_config.num_records - ycsb_config.starting_num_records;
        uint32_t want = num_inserts;
        uint32_t have = (free_start < free_total) ? (free_total - free_start) : 0;
        uint32_t take = std::min(want, have);

        if (take == 0 && want > 0)
        {
            logger.Error("Out of free rows for inserts! want: {}, have: {}", want, have);
            throw std::runtime_error("Out of free rows for inserts");
        }

        // assign crids to those keys
        auto zipped_inserts =
            thrust::make_zip_iterator(thrust::make_tuple(dp_valid_inserts, dp_free_rows + free_start));
        index->insert(zipped_inserts, zipped_inserts + take);
        free_start += take;
        logger.Trace("Free rows used: {}", free_start);
        // Now that the inserts are in there, we can translate every txn's key to a record_id for execution
        indexYcsbKernel<<<(ycsb_config.num_txns + block_size - 1) / block_size, block_size>>>(
            GpuTxnArray(txn_array), GpuTxnArray(index_array), index_view, ycsb_config.num_txns);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));

        // Mirror this epoch's INSERTs into the CPU shadow. D2H the
        // insert keys into the shadow's pinned host buffer, then
        // delegate the host-side mirror update to YcsbCpuShadowIndex.
        // The D2H is on the default stream so it implicitly drains the
        // GPU bulk_insert issued above.
        if (take > 0) {
            const uint32_t old_free_start = free_start - take;
            gpu_err_check(cudaMemcpy(
                shadow_.h_insert_keys(), d_valid_inserts,
                take * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            shadow_.mirrorEpoch(take, old_free_start);
        }

        // Delete application, after this epoch's lookups so every
        // transaction of the epoch still resolves the key (epoch-boundary
        // visibility); no transaction of the next epoch can. Emits the
        // op-ordered delete set, erases the keys from the cuco map, and
        // appends the keys to the durable delete log. The compacted CRID
        // list stays on device for the stager's reclaim-first marking.
        if (ycsb_config.txn_mix.num_deletes > 0) {
            const size_t n_slots =
                static_cast<size_t>(ycsb_config.num_txns) * ycsb_config.num_ops_per_txn;
            prepareYcsbDeleteKernel<<<(ycsb_config.num_txns + block_size - 1) / block_size, block_size>>>(
                GpuTxnArray(txn_array), GpuTxnArray(index_array),
                d_deletes, d_delete_crids, d_delete_flags, ycsb_config.num_txns);
            gpu_err_check(cudaPeekAtLastError());
            cub::DeviceSelect::Flagged(d_flagged_temp, flagged_temp_bytes,
                d_deletes, d_delete_flags, d_valid_deletes, d_num_delete, n_slots);
            cub::DeviceSelect::Flagged(d_flagged_temp, flagged_temp_bytes,
                d_delete_crids, d_delete_flags, d_valid_delete_crids, d_num_delete, n_slots);
            gpu_err_check(cudaStreamSynchronize(0));
            const uint32_t num_deletes = *h_num_delete;
            num_deletes_this_epoch = num_deletes;
            logger.Info("Found {} deletes", num_deletes);
            if (num_deletes > 0) {
                index->erase(dp_valid_deletes, dp_valid_deletes + num_deletes);
                gpu_err_check(cudaStreamSynchronize(0));
                gpu_err_check(cudaMemcpy(
                    shadow_.h_delete_keys(), d_valid_deletes,
                    num_deletes * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                shadow_.mirrorEpochDeletes(num_deletes, delete_count);
                delete_count += num_deletes;
            }
        }
    }

    void rebuildCucoFromShadow(uint32_t current_free_start, uint32_t current_delete_count)
    {
        auto& logger = Logger::GetInstance();

        // Safety: caller must ensure no pending GPU work references the old index_view.
        gpu_err_check(cudaStreamSynchronize(0));

        const uint64_t max_crid = static_cast<uint64_t>(ycsb_config.starting_num_records)
                                  + static_cast<uint64_t>(current_free_start);
        logger.Info("Rebuilding cuco from CPU shadow (max_crid={})", max_crid);

        // Tear down old cuco + create fresh one at the same load_factor,
        // with the same erased-key sentinel as the ctor (cuco erase throws
        // on a map built without it).
        index.reset();
        index = std::make_shared<YcsbIndexType>(
            static_cast<size_t>(std::ceil(ycsb_config.num_records / load_factor)),
            empty_key_sentinel, empty_value_sentinel, erased_key_sentinel);
        index_view = index->get_device_view();
        free_start = current_free_start;   // resync host scalar
        // Resync the delete-log cursor to the rollback point so replay's
        // re-applied deletes overwrite the log tail at the same positions.
        delete_count = current_delete_count;
        num_deletes_this_epoch = 0;

        // Reseed d_free_rows so d_free_rows[j] == starting_num_records + j,
        // matching loadInitialData()'s invariant. Idempotent overwrite; assumes
        // the GPU context itself is alive (d_free_rows pointer still valid).
        // Full context-loss recovery would need to reallocate this array first.
        const uint32_t remaining =
            ycsb_config.num_records - ycsb_config.starting_num_records;
        thrust::sequence(dp_free_rows,
                         dp_free_rows + remaining,
                         ycsb_config.starting_num_records);

        // Stream uploads in batches to cap device staging memory.
        constexpr size_t kBatch = 1u << 20; // 1 M entries per batch
        thrust::device_vector<uint32_t> d_keys(kBatch);
        thrust::device_vector<uint32_t> d_vals(kBatch);
        std::vector<uint32_t> h_keys;  h_keys.reserve(kBatch);
        std::vector<uint32_t> h_vals;  h_vals.reserve(kBatch);

        auto flush_batch = [&]() {
            if (h_keys.empty()) return;
            thrust::copy(h_keys.begin(), h_keys.end(), d_keys.begin());
            thrust::copy(h_vals.begin(), h_vals.end(), d_vals.begin());
            auto zipped = thrust::make_zip_iterator(thrust::make_tuple(
                d_keys.begin(), d_vals.begin()));
            index->insert(zipped, zipped + h_keys.size());
            h_keys.clear();
            h_vals.clear();
        };

        // Drop stragglers from epochs > E-2 (e.g. E-1 / E if we crashed
        // mid-E) so the shadow matches the rebuilt cuco exactly; replay
        // will repopulate them deterministically.
        shadow_.eraseStragglers(static_cast<uint32_t>(max_crid));

        // Walk the cleaned shards and batch-upload to the fresh cuco map.
        size_t total = 0;
        for (const auto& shard : shadow_.shards()) {
            for (const auto& kv : shard) {
                h_keys.push_back(kv.first);
                h_vals.push_back(kv.second);
                ++total;
                if (h_keys.size() == kBatch) flush_batch();
            }
        }
        flush_batch();

        // After rollback, the shadow holds no entry past the new
        // free_start, so reset its snapshots to match. The caller must
        // execute at least two more epochs before the next rollback is
        // valid.
        shadow_.syncSnapshotsToRollback(current_free_start);

        gpu_err_check(cudaStreamSynchronize(0));
        logger.Info("Rebuild complete: uploaded {} entries", total);
    }

};
} // namespace

YcsbGpuIndex::YcsbGpuIndex(YcsbConfig ycsb_config, YcsbCpuShadowIndex& shadow)
    : ycsb_config(ycsb_config)
{
    gpu_index_impl = std::make_any<YcsbGpuIndexImpl>(ycsb_config, shadow);
}
void YcsbGpuIndex::loadInitialData()
{
    auto &impl = std::any_cast<YcsbGpuIndexImpl &>(gpu_index_impl);
    impl.loadInitialData();
}

void YcsbGpuIndex::indexTxns(TxnArray<YcsbTxn> &txn_array, TxnArray<YcsbTxnParam> &index_array, uint32_t epoch_id)
{
    auto &impl = std::any_cast<YcsbGpuIndexImpl &>(gpu_index_impl);
    impl.indexTxns(txn_array, index_array, epoch_id);
}

void YcsbGpuIndex::rebuildCucoFromShadow(uint32_t current_free_start, uint32_t current_delete_count)
{
    auto &impl = std::any_cast<YcsbGpuIndexImpl &>(gpu_index_impl);
    impl.rebuildCucoFromShadow(current_free_start, current_delete_count);
}

uint32_t YcsbGpuIndex::getInsertCount() const
{
    auto const &impl = std::any_cast<YcsbGpuIndexImpl const &>(gpu_index_impl);
    return impl.free_start;
}

uint32_t YcsbGpuIndex::getDeleteCount() const
{
    auto const &impl = std::any_cast<YcsbGpuIndexImpl const &>(gpu_index_impl);
    return impl.delete_count;
}

const uint32_t* YcsbGpuIndex::deleteCridsDevice() const
{
    auto const &impl = std::any_cast<YcsbGpuIndexImpl const &>(gpu_index_impl);
    return impl.d_valid_delete_crids;
}

uint32_t YcsbGpuIndex::numDeletesThisEpoch() const
{
    auto const &impl = std::any_cast<YcsbGpuIndexImpl const &>(gpu_index_impl);
    return impl.num_deletes_this_epoch;
}
} // namespace epic::ycsb