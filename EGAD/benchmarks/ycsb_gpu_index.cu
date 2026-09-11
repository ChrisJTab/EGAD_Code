//
// Created by Shujian Qian on 2023-11-22.
//

#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>
#include <cuda/std/atomic>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <benchmarks/ycsb_gpu_index.h>
#include <benchmarks/tpcc_gpu_index.h>
#include <benchmarks/tpcc_table.h>
#include <benchmarks/epoch_timeline.cuh>
#include <gpu_txn.cuh>
#include <util_log.h>
#include <util_gpu_error_check.cuh>

#include <cuco/static_map.cuh>

#include <cub/cub.cuh>

namespace epic::ycsb {

namespace {

using YcsbIndexType = cuco::static_map<uint32_t, uint32_t>;
using YcsbIndexDeviceView = YcsbIndexType::device_view;
using epic::timeline::kAbsent;

struct IsNotSentinel {
    __device__ bool operator()(uint32_t x) const {
        return x != kAbsent; // matches what k_extract_ops writes for slots that are not of the kind
    }
};

// Predicate over slot indices: the slot holds a non-sentinel entry of the
// bound per-slot array. Used to compact slot indices in slot order
// alongside the key compaction that uses IsNotSentinel.
struct SlotHolds {
    const uint32_t* keys;
    __device__ bool operator()(uint32_t slot) const { return keys[slot] != kAbsent; }
};

// Per-slot extraction of the epoch's index events. Slot = tid * 10 + i, the
// operation's serial position. insert[slot] / del[slot] carry the key of an
// INSERT / DELETE op and kAbsent otherwise; ev[slot] is 1 for either.
void __global__ k_extract_ops(GpuTxnArray txns, uint32_t *insert, uint32_t *del, uint32_t *pos, uint32_t num_txns)
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
        const YcsbOpType op = txn->ops[i];
        const uint32_t key = txn->keys[i];
        insert[base + i] = (op == YcsbOpType::INSERT) ? key : kAbsent;
        del[base + i]    = (op == YcsbOpType::DELETE) ? key : kAbsent;
        if (pos) pos[base + i] = base + i;
    }
}

// The record a key resolves to on the fast path, before deletes are
// considered: the index lookup, except that a record minted this epoch is
// visible only from the slot of its insert onward.
__device__ __forceinline__ uint32_t fastLookup(YcsbIndexDeviceView index_view, uint32_t key, uint32_t slot,
                                               uint32_t minted_begin, uint32_t num_minted,
                                               const uint32_t* __restrict__ ins_slot)
{
    uint32_t rid = kAbsent;
    auto record_found = index_view.find(key);
    if (record_found != index_view.end())
    {
        rid = record_found->second.load(cuda::std::memory_order_relaxed);
    }
    if (rid != kAbsent && rid - minted_begin < num_minted && slot < ins_slot[rid - minted_begin]) rid = kAbsent;
    return rid;
}

// Fast delete path, pass 1: record the earliest slot at which each record
// is deleted this epoch (del_pos, per CRID, kAbsent when none).
void __global__ k_record_deletes(GpuTxnArray txns, YcsbIndexDeviceView index_view, uint32_t num_txns,
                                 uint32_t minted_begin, uint32_t num_minted, const uint32_t* __restrict__ ins_slot,
                                 uint32_t* __restrict__ del_pos)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns) return;
    YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(txns.getTxn(tid)->data);
    for (int i = 0; i < 10; ++i)
    {
        if (txn->ops[i] != YcsbOpType::DELETE) continue;
        const uint32_t slot = tid * 10 + i;
        const uint32_t rid = fastLookup(index_view, txn->keys[i], slot, minted_begin, num_minted, ins_slot);
        if (rid != kAbsent) atomicMin(&del_pos[rid], slot);
    }
}

// Fast delete path, pass 3: the effective deletes (the first delete of a
// live record), one flag per slot with the key and the ended CRID.
void __global__ k_effective_deletes(GpuTxnArray txns, GpuTxnArray index, uint32_t num_txns,
                                    const uint32_t* __restrict__ del_pos,
                                    uint32_t* __restrict__ keys, uint32_t* __restrict__ crids, uint8_t* __restrict__ flags)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns) return;
    YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(txns.getTxn(tid)->data);
    YcsbTxnParam *param = reinterpret_cast<YcsbTxnParam *>(index.getTxn(tid)->data);
    for (int i = 0; i < 10; ++i)
    {
        const uint32_t slot = tid * 10 + i;
        const uint32_t rid = param->record_ids[i];
        const bool eff = txn->ops[i] == YcsbOpType::DELETE && rid != kAbsent && del_pos[rid] == slot;
        flags[slot] = eff ? 1 : 0;
        if (eff) { keys[slot] = txn->keys[i]; crids[slot] = rid; }
    }
}

// Translate each op's key to its record id via the index, copying ops and
// field ids through to the executor-facing params. A key the index does not
// hold resolves to kAbsent. Fast path: a record minted this epoch is
// visible only to the operations at or after the slot of its insert, and a
// record deleted this epoch (del_pos, pass 1) only to the operations before
// its first delete; the delete itself resolves to the record it ends.
void __global__ indexYcsbKernel(GpuTxnArray txn, GpuTxnArray index, YcsbIndexDeviceView index_view, uint32_t num_txns,
                                uint32_t minted_begin, uint32_t num_minted, const uint32_t* __restrict__ ins_slot,
                                const uint32_t* __restrict__ del_pos)
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
        const uint32_t slot = static_cast<uint32_t>(tid * 10 + i);
        uint32_t rid = fastLookup(index_view, txn_ptr->keys[i], slot, minted_begin, num_minted, ins_slot);
        if (del_pos != nullptr && rid != kAbsent && del_pos[rid] < slot) rid = kAbsent;
        index_ptr->record_ids[i] = rid;
        index_ptr->ops[i] = (rid == kAbsent) ? YcsbOpType::NOOP : txn_ptr->ops[i];
        index_ptr->field_ids[i] = txn_ptr->fields[i];
    }
}

// General path (delete events this epoch): the epoch-start lookup is the
// fallback for keys without events; keys with events resolve against their
// timeline at the op's slot.
void __global__ indexYcsbTimelineKernel(GpuTxnArray txn, GpuTxnArray index, YcsbIndexDeviceView index_view,
                                        uint32_t num_txns, epic::timeline::TimelineView<32> tl)
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
        const uint32_t key = txn_ptr->keys[i];
        uint32_t rid = kAbsent;
        auto record_found = index_view.find(key);
        if (record_found != index_view.end())
        {
            rid = record_found->second.load(cuda::std::memory_order_relaxed);
        }
        const uint32_t slot = static_cast<uint32_t>(tid * 10 + i);
        rid = tl.resolve(key, slot, slot, txn_ptr->ops[i] == YcsbOpType::INSERT, rid);
        index_ptr->record_ids[i] = rid;
        index_ptr->ops[i] = (rid == kAbsent) ? YcsbOpType::NOOP : txn_ptr->ops[i];
        index_ptr->field_ids[i] = txn_ptr->fields[i];
    }
}

class YcsbGpuIndexImpl
{
public:
    static constexpr double load_factor = 0.8;
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
    // pinned host buffers for D2H of insert and delete keys) lives in
    // YcsbBenchmark as a YcsbCpuShadowIndex. This class holds a reference.
    // The shadow is the host-side ground truth: after a crash, recovery
    // repopulates a fresh GPU index from it. Declared here (next to
    // ycsb_config, before the cuco map) so the ctor's member-init list order
    // matches declaration order.
    YcsbCpuShadowIndex& shadow_;

    std::shared_ptr<YcsbIndexType> index;
    YcsbIndexDeviceView index_view;

    uint32_t *d_free_rows;
    thrust::device_ptr<uint32_t> dp_free_rows;
    uint32_t free_start = 0;

    // Per-slot op columns (n_slots = num_txns * ops per txn): the key of an
    // INSERT / DELETE op at that slot (kAbsent otherwise) and the event flag.
    uint32_t *d_inserts = nullptr, *d_valid_inserts = nullptr;
    uint32_t *d_deletes = nullptr, *d_valid_deletes = nullptr;
    thrust::device_ptr<uint32_t> dp_inserts, dp_valid_inserts, dp_valid_deletes;
    // Slot of the j-th minted insert (slot order); the visibility bound of
    // the record minted this epoch as CRID minted_begin + j.
    uint32_t *d_ins_slot = nullptr;
    uint32_t *d_num_insert;      // device-accessible pointer (mapped)
    uint32_t *h_num_insert;      // host-accessible pointer (mapped, same physical memory)
    uint32_t *d_num_delete = nullptr;   // mapped device pointer
    uint32_t *h_num_delete = nullptr;   // mapped host pointer (same memory)
    uint32_t minted_begin = 0;           // first CRID minted this epoch
    uint32_t num_minted_this_epoch = 0;
    uint32_t delete_count = 0;           // cumulative; the durable delete-log cursor
    uint32_t num_deletes_this_epoch = 0;
    // Fast delete path (delete-bearing mixes without a rejected insert):
    // del_pos[crid] = the earliest slot deleting the record this epoch
    // (kAbsent when none), allocated on first use, reset per epoch for the
    // recorded records; per-slot effective-delete columns and their
    // compactions (entry order, the delete log's order).
    uint32_t *d_del_pos = nullptr;
    uint32_t *d_edel_crid = nullptr;
    uint8_t  *d_edel_flag = nullptr;
    uint32_t *d_valid_delete_crids = nullptr;   // also the insert check's scratch, which runs before the delete pass
    const uint32_t *d_edel_out = nullptr;       // this epoch's effective-delete CRIDs (fast or timeline path)
    // Once the map has erased anything, the fast path checks its inserts
    // against the map itself (see epic::timeline::insertFresh).
    bool index_has_erased = false;
#ifdef EGAD_VALIDATION
    bool order_blind_control = false;    // resolve order-blind on both paths; the oracle must reject it
#endif

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // === Timeline (general path) ===
    // Built at construction for delete-bearing mixes and on first use
    // otherwise (a duplicate or existing key among an epoch's inserts sends
    // that epoch through the general path). Entry = slot; PosBits = 32.
    uint32_t num_slots = 0;
    epic::timeline::EpochTimeline<uint32_t, 32> timeline;
    void ensureTimeline() { timeline.ensure(num_slots, num_slots, num_slots - 1); }

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
        num_slots = static_cast<uint32_t>(ycsb_config.num_txns * ycsb_config.num_ops_per_txn);
        // Per-slot op columns and their compactions, sized to the worst case
        // (every op an INSERT, or every op a DELETE).
        gpu_err_check(cudaMalloc(&d_inserts, sizeof(uint32_t) * num_slots));
        gpu_err_check(cudaMalloc(&d_valid_inserts, sizeof(uint32_t) * num_slots));
        gpu_err_check(cudaMalloc(&d_deletes, sizeof(uint32_t) * num_slots));
        gpu_err_check(cudaMalloc(&d_valid_deletes, sizeof(uint32_t) * num_slots));
        gpu_err_check(cudaMalloc(&d_valid_delete_crids, sizeof(uint32_t) * num_slots));
        gpu_err_check(cudaMalloc(&d_edel_crid, sizeof(uint32_t) * num_slots));
        gpu_err_check(cudaMalloc(&d_edel_flag, sizeof(uint8_t) * num_slots));
        gpu_err_check(cudaMalloc(&d_ins_slot, sizeof(uint32_t) * num_slots));
        dp_inserts = thrust::device_pointer_cast(d_inserts);
        dp_valid_inserts = thrust::device_pointer_cast(d_valid_inserts);
        dp_valid_deletes = thrust::device_pointer_cast(d_valid_deletes);
        gpu_err_check(cudaHostAlloc(&h_num_insert, sizeof(uint32_t), cudaHostAllocMapped));
        gpu_err_check(cudaHostGetDevicePointer(&d_num_insert, h_num_insert, 0));
        gpu_err_check(cudaHostAlloc(&h_num_delete, sizeof(uint32_t), cudaHostAllocMapped));
        *h_num_delete = 0;
        gpu_err_check(cudaHostGetDevicePointer(&d_num_delete, h_num_delete, 0));
        // Temp storage for the cub compactions: one allocation covering the
        // largest of the calls this class issues (all over n_slots items).
        size_t need = 0, bytes = 0;
        IsNotSentinel pred{};
        cub::DeviceSelect::If(nullptr, bytes, dp_inserts, dp_valid_inserts, d_num_insert, num_slots, pred);
        need = std::max(need, bytes); bytes = 0;
        SlotHolds holds{d_inserts};
        cub::DeviceSelect::If(nullptr, bytes, thrust::counting_iterator<uint32_t>(0), d_ins_slot, d_num_insert, num_slots, holds);
        need = std::max(need, bytes); bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, bytes, d_deletes, d_edel_flag, d_valid_deletes, d_num_delete, num_slots);
        need = std::max(need, bytes); bytes = 0;
        temp_storage_bytes = need;
        logger.Trace("Allocating {} bytes for temp storage", formatSizeBytes(temp_storage_bytes));
        gpu_err_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        if (ycsb_config.txn_mix.num_deletes > 0) {
            ensureTimeline();
        }
#ifdef EGAD_VALIDATION
        // Validation control: both paths resolve order-blind, as the
        // epoch-boundary semantics did; the oracle must reject it.
        if (std::getenv("EPIC_TIMELINE_ORDER_BLIND") != nullptr) {
            order_blind_control = true;
            timeline.setOrderBlind(true);
            logger.Info("[TIMELINE] ORDER-BLIND control active");
        }
#endif

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
        // E-1 and E-2 before any of this epoch's inserts increment the
        // cursor.
        shadow_.shiftSnapshotsAtEpochStart(free_start);

        constexpr uint32_t block_size = 512;
        const uint32_t n_slots = num_slots;
        const uint32_t txn_blocks = (ycsb_config.num_txns + block_size - 1) / block_size;

        k_extract_ops<<<txn_blocks, block_size>>>(GpuTxnArray(txn_array), d_inserts, d_deletes, nullptr,
                                                  ycsb_config.num_txns);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));

        num_deletes_this_epoch = 0;
        num_minted_this_epoch = 0;
        minted_begin = ycsb_config.starting_num_records + free_start;

        // Every mix starts on the fast path (bulk insert; deletes through
        // del_pos); an epoch whose inserts are rejected resolves through the
        // timeline instead.
        bool general = false;
#ifdef EGAD_VALIDATION
        // Validation hook: resolve every epoch through the timeline, so the
        // workloads without deletes exercise it and must reproduce the fast
        // path's results exactly.
        static const bool kForceGeneral = std::getenv("EPIC_TIMELINE_FORCE_GENERAL") != nullptr;
        if (kForceGeneral) general = true;
#endif
        if (!general)
        {
            // Fast path: the epoch's inserts create records, every other op
            // resolves on the epoch-start index plus those records, minus
            // the records deleted before the op's slot (del_pos). The
            // minting rule is the j-th INSERT in slot order.
            IsNotSentinel pred{};
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, dp_inserts, dp_valid_inserts, d_num_insert,
                n_slots, pred);
            SlotHolds holds{d_inserts};
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, thrust::counting_iterator<uint32_t>(0), d_ins_slot,
                d_num_insert, n_slots, holds);
            gpu_err_check(cudaStreamSynchronize(0));
            const uint32_t num_inserts = *h_num_insert;  // read directly from mapped memory, no D2H needed
            logger.Info("Found {} inserts", num_inserts);
            checkFreeRows(num_inserts);
            if (num_inserts > 0)
            {
                // Every insert must be of a key the index does not hold, once
                // each in the epoch. An insert of a live key, or a second
                // insert of the same new key, sends the epoch through the
                // timeline, which resolves both in serial order and mints only
                // for the life-creating inserts.
                thrust::device_ptr<uint32_t> scratch(d_valid_delete_crids);
                if (epic::timeline::insertFresh(*index, index_has_erased, dp_valid_inserts, dp_free_rows + free_start,
                                                num_inserts, minted_begin, scratch))
                {
                    num_minted_this_epoch = num_inserts;
                }
                else
                {
                    logger.Info("Epoch {}: an insert is not of a fresh key, resolving through the timeline", epoch_id);
                    if (epic::timeline::eraseTentative(*index, dp_valid_inserts, num_inserts, minted_begin) > 0) {
                        index_has_erased = true;
                    }
                    general = true;
                }
            }
            if (!general)
            {
                const bool deletes = ycsb_config.txn_mix.num_deletes > 0;
                if (deletes) {
                    ensureDelPos();
                    k_record_deletes<<<txn_blocks, block_size>>>(GpuTxnArray(txn_array), index_view, ycsb_config.num_txns,
                        minted_begin, num_minted_this_epoch, d_ins_slot, d_del_pos);
                    gpu_err_check(cudaPeekAtLastError());
                }
                uint32_t visible_minted = num_minted_this_epoch;
                const uint32_t* lookup_del_pos = deletes ? d_del_pos : nullptr;
#ifdef EGAD_VALIDATION
                if (order_blind_control) { visible_minted = 0; lookup_del_pos = nullptr; }
#endif
                indexYcsbKernel<<<txn_blocks, block_size>>>(GpuTxnArray(txn_array), GpuTxnArray(index_array),
                    index_view, ycsb_config.num_txns, minted_begin, visible_minted, d_ins_slot, lookup_del_pos);
                gpu_err_check(cudaPeekAtLastError());
                if (deletes) {
                    k_effective_deletes<<<txn_blocks, block_size>>>(GpuTxnArray(txn_array), GpuTxnArray(index_array),
                        ycsb_config.num_txns, d_del_pos, d_deletes, d_edel_crid, d_edel_flag);
                    gpu_err_check(cudaPeekAtLastError());
                    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_deletes, d_edel_flag, d_valid_deletes,
                        d_num_delete, n_slots);
                    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_edel_crid, d_edel_flag,
                        d_valid_delete_crids, d_num_delete, n_slots);
                    gpu_err_check(cudaStreamSynchronize(0));
                    num_deletes_this_epoch = *h_num_delete;
                    d_edel_out = d_valid_delete_crids;
                    if (num_deletes_this_epoch > 0) {
                        epic::timeline::k_reset_positions<<<(num_deletes_this_epoch + 255) / 256, 256>>>(
                            d_valid_delete_crids, num_deletes_this_epoch, d_del_pos);
                        gpu_err_check(cudaPeekAtLastError());
                        index->erase(dp_valid_deletes, dp_valid_deletes + num_deletes_this_epoch);
                        index_has_erased = true;
                    }
                }
                gpu_err_check(cudaStreamSynchronize(0));
            }
        }
        if (general)
        {
            resolveThroughTimeline(txn_array, index_array, epoch_id);
            d_edel_out = timeline.edelCrids();
        }

        const uint32_t take = num_minted_this_epoch;
        free_start += take;
        logger.Trace("Free rows used: {}", free_start);

        // Mirror this epoch's minted inserts into the CPU shadow's durable
        // insert log: D2H the keys (slot order, j-th key <-> CRID
        // minted_begin + j) into the shadow's pinned host buffer, then
        // delegate the append to YcsbCpuShadowIndex. The D2H is on the
        // default stream so it drains the GPU work issued above.
        if (take > 0) {
            const uint32_t old_free_start = free_start - take;
            gpu_err_check(cudaMemcpy(
                shadow_.h_insert_keys(), general ? timeline.mintedKeys() : d_valid_inserts,
                take * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            shadow_.mirrorEpoch(take, old_free_start);
        }

        // Effective deletes: the keys leave the index at the epoch boundary
        // (done inside the timeline finalize), their (key, CRID) pairs go to
        // the durable delete log, and the CRID list stays on device for the
        // stager's reclaim-first marking.
        if (num_deletes_this_epoch > 0) {
            logger.Info("Found {} deletes", num_deletes_this_epoch);
            gpu_err_check(cudaMemcpy(
                shadow_.h_delete_keys(), general ? timeline.edelKeys() : d_valid_deletes,
                num_deletes_this_epoch * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            gpu_err_check(cudaMemcpy(
                shadow_.h_delete_crids(), d_edel_out,
                num_deletes_this_epoch * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            shadow_.mirrorEpochDeletes(num_deletes_this_epoch, delete_count);
            delete_count += num_deletes_this_epoch;
        } else if (ycsb_config.txn_mix.num_deletes > 0) {
            logger.Info("Found 0 deletes");
        }
    }

    void ensureDelPos()
    {
        if (d_del_pos) return;
        gpu_err_check(cudaMalloc(&d_del_pos, sizeof(uint32_t) * ycsb_config.num_records));
        gpu_err_check(cudaMemset(d_del_pos, 0xff, sizeof(uint32_t) * ycsb_config.num_records));
    }

    void checkFreeRows(uint32_t want)
    {
        const uint32_t free_total = ycsb_config.num_records - ycsb_config.starting_num_records;
        const uint32_t have = (free_start < free_total) ? (free_total - free_start) : 0;
        if (want > have)
        {
            Logger::GetInstance().Error("Out of free rows for inserts! want: {}, have: {}", want, have);
            throw std::runtime_error("Out of free rows for inserts");
        }
    }

    // General path: build the epoch's timelines, mint CRIDs for the
    // life-creating inserts, resolve every op at its slot, then apply each
    // key's final state to the index.
    void resolveThroughTimeline(TxnArray<YcsbTxn> &txn_array, TxnArray<YcsbTxnParam> &index_array, uint32_t epoch_id)
    {
        auto &logger = Logger::GetInstance();
        ensureTimeline();
        constexpr uint32_t block_size = 512;
        const uint32_t txn_blocks = (ycsb_config.num_txns + block_size - 1) / block_size;

        k_extract_ops<<<txn_blocks, block_size>>>(GpuTxnArray(txn_array), timeline.insKeys(), timeline.delKeys(),
                                                  timeline.pos(), ycsb_config.num_txns);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));
        timeline.build([this](const uint32_t* keys, uint32_t n, uint32_t* out) {
                thrust::device_ptr<const uint32_t> k(keys);
                thrust::device_ptr<uint32_t> o(out);
                index->find(k, k + n, o);
            }, minted_begin);
        const uint32_t num_minted = timeline.numMinted();
        logger.Info("Found {} inserts", num_minted);
        checkFreeRows(num_minted);
        num_minted_this_epoch = num_minted;

        // Resolve every op: epoch-start lookup, overridden by the timeline
        // for keys with events.
        indexYcsbTimelineKernel<<<txn_blocks, block_size>>>(GpuTxnArray(txn_array), GpuTxnArray(index_array), index_view,
            ycsb_config.num_txns, timeline.view());
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaStreamSynchronize(0));

        num_deletes_this_epoch = timeline.numEffectiveDeletes();
        // Index update at the epoch boundary: erase the keys that end absent
        // or replaced, then insert the final lives minted this epoch.
        const uint32_t n_erase = timeline.numErase(), n_ins = timeline.numInsert();
        if (n_erase > 0) {
            thrust::device_ptr<const uint32_t> k(timeline.eraseKeys());
            index->erase(k, k + n_erase);
            index_has_erased = true;
        }
        if (n_ins > 0) {
            thrust::device_ptr<const uint32_t> k(timeline.insertKeys());
            thrust::device_ptr<const uint32_t> v(timeline.insertCrids());
            auto zipped = thrust::make_zip_iterator(thrust::make_tuple(k, v));
            index->insert(zipped, zipped + n_ins);
        }
        gpu_err_check(cudaStreamSynchronize(0));
        logger.Trace("Epoch {} timeline: events={} minted={} effective_deletes={} erased={} inserted={}",
                     epoch_id, timeline.numEvents(), num_minted, num_deletes_this_epoch, n_erase, n_ins);
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
        index_has_erased = false;          // a fresh map has no erased slot
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
    return impl.d_edel_out;
}

uint32_t YcsbGpuIndex::numDeletesThisEpoch() const
{
    auto const &impl = std::any_cast<YcsbGpuIndexImpl const &>(gpu_index_impl);
    return impl.num_deletes_this_epoch;
}

uint32_t YcsbGpuIndex::mintedBegin() const
{
    auto const &impl = std::any_cast<YcsbGpuIndexImpl const &>(gpu_index_impl);
    return impl.minted_begin;
}

uint32_t YcsbGpuIndex::numMintedThisEpoch() const
{
    auto const &impl = std::any_cast<YcsbGpuIndexImpl const &>(gpu_index_impl);
    return impl.num_minted_this_epoch;
}

#ifdef EGAD_VALIDATION
uint32_t YcsbGpuIndex::verifyLiveMapping(const std::vector<uint32_t>& keys, const std::vector<uint32_t>& expected,
                                         const std::vector<uint32_t>& dead) const
{
    auto const &impl = std::any_cast<YcsbGpuIndexImpl const &>(gpu_index_impl);
    auto &logger = Logger::GetInstance();
    uint32_t mismatches = 0, dead_found = 0;
    constexpr size_t kBatch = 1u << 20;
    thrust::device_vector<uint32_t> d_keys(kBatch), d_out(kBatch);
    std::vector<uint32_t> h_out(kBatch);
    for (size_t off = 0; off < keys.size(); off += kBatch) {
        const size_t n = std::min(kBatch, keys.size() - off);
        thrust::copy(keys.begin() + off, keys.begin() + off + n, d_keys.begin());
        impl.index->find(d_keys.begin(), d_keys.begin() + n, d_out.begin());
        thrust::copy(d_out.begin(), d_out.begin() + n, h_out.begin());
        for (size_t i = 0; i < n; ++i) {
            if (h_out[i] != expected[off + i]) {
                if (mismatches < 10) {
                    logger.Error("[MAP-CHECK] key {} expected {} found {}", keys[off + i], expected[off + i],
                                 static_cast<int64_t>(h_out[i] == kAbsent ? -1 : static_cast<int64_t>(h_out[i])));
                }
                ++mismatches;
            }
        }
    }
    for (size_t off = 0; off < dead.size(); off += kBatch) {
        const size_t n = std::min(kBatch, dead.size() - off);
        thrust::copy(dead.begin() + off, dead.begin() + off + n, d_keys.begin());
        impl.index->find(d_keys.begin(), d_keys.begin() + n, d_out.begin());
        thrust::copy(d_out.begin(), d_out.begin() + n, h_out.begin());
        for (size_t i = 0; i < n; ++i) {
            if (h_out[i] != kAbsent) {
                if (dead_found < 10) logger.Error("[MAP-CHECK] dead key {} still maps to {}", dead[off + i], h_out[i]);
                ++dead_found;
            }
        }
    }
    const uint32_t violations = mismatches + dead_found;
    if (violations == 0) {
        logger.Info("[MAP-CHECK] PASS ycsb live={} dead_probed={} (index agrees with the serial-order model)",
                    keys.size(), dead.size());
    } else {
        logger.Error("[MAP-CHECK] FAILED ycsb live={} mismatches={} dead_probed={} dead_found={}",
                     keys.size(), mismatches, dead.size(), dead_found);
    }
    return violations;
}
#endif // EGAD_VALIDATION

} // namespace epic::ycsb
