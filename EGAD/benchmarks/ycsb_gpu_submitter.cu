//
// Created by Shujian Qian on 2023-11-22.
//

#include <thrust/for_each.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>

#include <benchmarks/ycsb_gpu_submitter.h>
#include <gpu_txn.cuh>
#include <benchmarks/ycsb_config.h>
#include <execution_planner.h>

namespace epic::ycsb {

namespace {

    // Copy device scalars to mapped host memory via a 1-thread kernel.
    // Bypasses the D2H copy engine — uses SM store to mapped memory instead.
    __global__ void k_copy_scalars_to_mapped(const uint32_t* src0, uint32_t* dst0,
                                              const uint32_t* src1, uint32_t* dst1,
                                              const uint32_t* src2, uint32_t* dst2) {
        if (threadIdx.x == 0) {
            if (src0 && dst0) *dst0 = *src0;
            if (src1 && dst1) *dst1 = *src1;
            if (src2 && dst2) *dst2 = *src2;
        }
    }

    void __global__ countReadWriteInsertKeys(const uint32_t* read_keys,
                                             const uint32_t* write_keys,
                                             const uint32_t* insert_keys,
                                             uint32_t n,
                                             uint32_t* d_read_count,
                                             uint32_t* d_write_count,
                                             uint32_t* d_insert_count)
    {
        constexpr uint32_t SENT = 0xffffffffu;
        uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n) return;

        if (read_keys[tid]   != SENT) atomicAdd(d_read_count, 1u);
        if (write_keys[tid]  != SENT) atomicAdd(d_write_count, 1u);
        if (insert_keys[tid] != SENT) atomicAdd(d_insert_count, 1u);
    }

/*
 * This kernel computes for each transaction, how many low-level operations it will expand into during execution planning
 * and stores that count into a device array
 */
void __global__ prepareSubmitYcsbTxn(YcsbConfig config, GpuTxnArray txns, uint32_t *num_ops, uint32_t num_txns)
{

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *base_txn_ptr = txns.getTxn(tid);
    YcsbTxnParam *txn = reinterpret_cast<YcsbTxnParam *>(base_txn_ptr->data);
    uint32_t ops = 0;
    for (auto &op : txn->ops)
    {
        switch (op)
        {
        case YcsbOpType::READ:
            ops += 1;
            break;
        case YcsbOpType::UPDATE:
            ops += config.split_field ? 1 : 2; /* if not split, update is RMW */
            break;
        case YcsbOpType::READ_MODIFY_WRITE:
            ops += 2;
            break;
        case YcsbOpType::FULL_READ:
            ops += config.split_field ? 10 : 1;
            break;
        case YcsbOpType::FULL_READ_MODIFY_WRITE:
            ops += config.split_field ? 11 : 2;
            break;
        case YcsbOpType::INSERT:
            ops += config.split_field ? 10 : 1;
            break;
        }
    }
    num_ops[tid] = ops;
}




/*
 * This kernel materializes the flat ops array for the next stage(execution planner), expanding each transaction's 10 YCSB
 * ops into primitive read/write operations and writing them contiguously during a precomputed per-txn offset
 * For each of the 10 logical ops in a txn, it emits the corresponding low-level ops based on the op type and config
 * The emitted op encodes:
 *  - the key to touch (record ID or recordID * 10 + fieldID if split)
 *  - the txn ID
 *  - the op type (read or write)
 *  - the location in the execution plan to write the result or read source from
 *
 *  Expansion rules:
 *  READ -> 1 read op
 *  UPDATE -> 1 write op (if split) or 1 read + 1 write op (if not split)
 *  READ_MODIFY_WRITE -> 1 read + 1 write op
 *  FULL_READ -> 10 read ops (if split) or 1 read op (if not split)
 *  FULL_READ_MODIFY_WRITE -> 10 read ops + 1 write op (if split) or 1 read + 1 write op (if not split)
 *  INSERT -> 10 write ops (if split) or 1 write op (if not split)
 */
// Post-submit pass: derive all_keys / read_keys / write_keys / insert_keys from
// the canonical ops[] array (and the parallel per-op is_insert flag emitted by
// submitYcsbTxn). Indexed by op_idx (not tid), so a warp of 32 threads touches
// 32 contiguous positions in each output array -- one coalesced 128B store per
// array per warp, vs ~32 scattered stores per array in the old per-txn emit.
// All four key arrays use 0xffffffffu as a sentinel for "this op doesn't
// belong in this set", matching the convention consumed by the hybrid stager.
void __global__ deriveStagerKeys(const op_t* ops, const uint8_t* op_is_insert, uint32_t n,
                                 uint32_t* all_keys,
                                 uint32_t* read_keys,
                                 uint32_t* write_keys,
                                 uint32_t* insert_keys)
{
    constexpr uint32_t SENT = 0xffffffffu;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    op_t op = ops[i];
    uint32_t key   = static_cast<uint32_t>(GET_RECORD_ID(op));
    bool is_write  = (GET_R_W(op) == write_op);
    bool is_insert = op_is_insert[i] != 0;

    all_keys[i]    = key;
    read_keys[i]   = is_write ? SENT : key;
    write_keys[i]  = is_write ? key  : SENT;
    insert_keys[i] = is_insert ? key : SENT;
}

// The stager-only arrays (all_keys / read_keys / write_keys) are populated by
// deriveStagerKeys (above) after this kernel completes. Writing them here would
// cost 3 extra uncoalesced stores per emit (adjacent txn threads have different
// offsets, so each store scatters across cache lines). The post-hoc derive kernel
// is indexed by op_idx instead of tid, so its accesses are fully coalesced.
void __global__ submitYcsbTxn(YcsbConfig config, GpuTxnArray txns, uint32_t *offset,
                              op_t *ops, uint8_t *op_is_insert, uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *base_txn_ptr = txns.getTxn(tid);
    YcsbTxnParam *txn = reinterpret_cast<YcsbTxnParam *>(base_txn_ptr->data);
    uint32_t op_idx = offset[tid];

    auto emit = [&](uint32_t key, uint32_t loc_off_words, uint8_t kind, bool is_insert){
        assert(kind == read_op || kind == write_op);
        ops[op_idx] = CREATE_OP(key, tid, kind, loc_off_words);
        // Always overwrite — the array is sized to max_num_ops and reused
        // across epochs. Stale entries past op_idx are never read by the
        // post-pass (which uses curr_num_ops as its bound).
        op_is_insert[op_idx] = is_insert ? 1 : 0;
        ++op_idx;
    };

    for (int i = 0; i < 10; ++i)
    {
        const bool op_is_insert_kind = (txn->ops[i] == YcsbOpType::INSERT);
        switch (txn->ops[i])
        {
        case YcsbOpType::READ:
            emit(config.split_field ? txn->record_ids[i] * 10 + txn->field_ids[i] : txn->record_ids[i],
                offsetof(YcsbExecPlan, plans[i].read_plan.read_loc) / sizeof(uint32_t),
                read_op, op_is_insert_kind);
            break;
        case YcsbOpType::UPDATE:
            if (config.split_field)
            {
                emit(txn->record_ids[i] * 10 + txn->field_ids[i],
                    offsetof(YcsbExecPlan, plans[i].update_plan.write_loc) / sizeof(uint32_t),
                    write_op, op_is_insert_kind);
            }
            else
            {
                emit(txn->record_ids[i],
                    offsetof(YcsbExecPlan, plans[i].copy_update_plan.read_loc) / sizeof(uint32_t),
                    read_op, op_is_insert_kind);
                emit(txn->record_ids[i],
                    offsetof(YcsbExecPlan, plans[i].copy_update_plan.write_loc) / sizeof(uint32_t),
                    write_op, op_is_insert_kind);
            }
            break;
        case YcsbOpType::READ_MODIFY_WRITE:
            emit(config.split_field ? txn->record_ids[i] * 10 + txn->field_ids[i] : txn->record_ids[i],
                offsetof(YcsbExecPlan, plans[i].read_modify_write_plan.read_loc) / sizeof(uint32_t),
                read_op, op_is_insert_kind);
            emit(config.split_field ? txn->record_ids[i] * 10 + txn->field_ids[i] : txn->record_ids[i],
                offsetof(YcsbExecPlan, plans[i].read_modify_write_plan.write_loc) / sizeof(uint32_t),
                write_op, op_is_insert_kind);
            break;
        case YcsbOpType::FULL_READ:
            if (config.split_field)
            {
                for (int j = 0; j < 10; ++j)
                {
                    emit(txn->record_ids[i] * 10 + j,
                        offsetof(YcsbExecPlan, plans[i].full_read_plan.read_locs[j]) / sizeof(uint32_t),
                        read_op, op_is_insert_kind);
                }
            }
            else
            {
                emit(txn->record_ids[i],
                    offsetof(YcsbExecPlan, plans[i].read_plan.read_loc) / sizeof(uint32_t),
                    read_op, op_is_insert_kind);
            }
            break;
        case YcsbOpType::FULL_READ_MODIFY_WRITE:
            if (config.split_field)
            {
                for (int j = 0; j < 10; ++j)
                {
                    emit(txn->record_ids[i] * 10 + j,
                        offsetof(YcsbExecPlan, plans[i].full_read_modify_write_plan.read_locs[j]) / sizeof(uint32_t),
                        read_op, op_is_insert_kind);
                }
                emit(txn->record_ids[i] * 10 + txn->field_ids[i],
                    offsetof(YcsbExecPlan, plans[i].full_read_modify_write_plan.write_loc) / sizeof(uint32_t),
                    write_op, op_is_insert_kind);
            }
            else
            {
                emit(txn->record_ids[i],
                    offsetof(YcsbExecPlan, plans[i].read_modify_write_plan.read_loc) / sizeof(uint32_t),
                        read_op, op_is_insert_kind);
                emit(txn->record_ids[i],
                    offsetof(YcsbExecPlan, plans[i].read_modify_write_plan.write_loc) / sizeof(uint32_t),
                    write_op, op_is_insert_kind);
            }
            break;
        case YcsbOpType::INSERT:
            if (config.split_field)
            {
                for (int j = 0; j < 10; ++j)
                {
                    emit(txn->record_ids[i] * 10 + j,
                        offsetof(YcsbExecPlan, plans[i].insert_full_plan.write_locs[j]) / sizeof(uint32_t),
                        write_op, op_is_insert_kind);
                }
            }
            else
            {
                emit(txn->record_ids[i],
                    offsetof(YcsbExecPlan, plans[i].insert_plan.write_loc) / sizeof(uint32_t),
                    write_op, op_is_insert_kind);
            }
            break;
        }
    }
}

} // namespace

YcsbGpuSubmitter::YcsbGpuSubmitter(TableSubmitDest submit_dest, YcsbConfig config)
    : YcsbSubmitter(submit_dest, config)
{
    gpu_err_check(cudaMalloc(&d_read_count_, sizeof(uint32_t)));
    gpu_err_check(cudaMalloc(&d_write_count_, sizeof(uint32_t)));
    gpu_err_check(cudaMalloc(&d_insert_count_, sizeof(uint32_t)));
    // Mapped scalars for copy-engine-free readback
    gpu_err_check(cudaHostAlloc(&h_mapped_, 4 * sizeof(uint32_t), cudaHostAllocMapped));
    memset(h_mapped_, 0, 4 * sizeof(uint32_t));
    gpu_err_check(cudaHostGetDevicePointer(&d_mapped_num_ops_,    &h_mapped_[0], 0));
    gpu_err_check(cudaHostGetDevicePointer(&d_mapped_read_cnt_,   &h_mapped_[1], 0));
    gpu_err_check(cudaHostGetDevicePointer(&d_mapped_write_cnt_,  &h_mapped_[2], 0));
    gpu_err_check(cudaHostGetDevicePointer(&d_mapped_insert_cnt_, &h_mapped_[3], 0));
}

YcsbGpuSubmitter::~YcsbGpuSubmitter()
{
    if (d_read_count_)   { cudaFree(d_read_count_);   d_read_count_   = nullptr; }
    if (d_write_count_)  { cudaFree(d_write_count_);  d_write_count_  = nullptr; }
    if (d_insert_count_) { cudaFree(d_insert_count_); d_insert_count_ = nullptr; }
    if (h_mapped_)       { cudaFreeHost(h_mapped_);   h_mapped_ = nullptr; }
}

void YcsbGpuSubmitter::submit(TxnArray<YcsbTxnParam> &txn_array)
{
    auto &logger = Logger::GetInstance();

    constexpr uint32_t block_size = 512;
    uint32_t num_blocks = (config.num_txns + block_size - 1) / block_size;
    prepareSubmitYcsbTxn<<<num_blocks, block_size>>>(
        config, GpuTxnArray(txn_array), submit_dest.d_num_ops, config.num_txns);
    gpu_err_check(cudaGetLastError());

    gpu_err_check(cub::DeviceScan::InclusiveSum(submit_dest.temp_storage, submit_dest.temp_storage_bytes,
        submit_dest.d_num_ops, submit_dest.d_op_offsets + 1, txn_array.num_txns));
    // Copy num_ops from device array to mapped memory via 1-thread kernel (bypasses copy engine)
    k_copy_scalars_to_mapped<<<1, 1>>>(
        submit_dest.d_op_offsets + txn_array.num_txns, d_mapped_num_ops_,
        nullptr, nullptr, nullptr, nullptr);

    // The per-txn submitter no longer writes the stager-only key arrays; they are
    // derived in a coalesced post-pass below (deriveStagerKeys). That pass only
    // runs in HYBRID_STAGING mode -- gpu_only / cpu_only don't need those arrays.
    bool hybrid = (config.execution_mode == ExecMode::HYBRID_STAGING);
    submitYcsbTxn<<<num_blocks, block_size>>>(config, GpuTxnArray(txn_array), submit_dest.d_op_offsets,
        reinterpret_cast<op_t *>(submit_dest.d_submitted_ops),
        submit_dest.d_op_is_insert,
        config.num_txns);

    gpu_err_check(cudaGetLastError());

    // Sync to get num_ops from mapped memory (copy_scalar kernel already wrote it)
    gpu_err_check(cudaStreamSynchronize(0));
    submit_dest.curr_num_ops = h_mapped_[0];

    // Only the hybrid stager consumes read/write/insert key counts; skip the
    // extra kernels + sync + mapped readbacks in gpu_only / cpu_only modes.
    if (config.execution_mode == ExecMode::HYBRID_STAGING) {
        // Derive all_keys / read_keys / write_keys / insert_keys from ops[] +
        // the parallel op_is_insert flag with fully coalesced accesses. Runs in
        // the default stream, so it's sequenced after submitYcsbTxn and before
        // countReadWriteInsertKeys below (which reads the arrays this kernel
        // writes).
        uint32_t derive_blocks = (submit_dest.curr_num_ops + block_size - 1) / block_size;
        deriveStagerKeys<<<derive_blocks, block_size>>>(
            reinterpret_cast<const op_t *>(submit_dest.d_submitted_ops),
            submit_dest.d_op_is_insert,
            submit_dest.curr_num_ops,
            submit_dest.d_all_keys,
            submit_dest.d_read_keys,
            submit_dest.d_write_keys,
            submit_dest.d_insert_keys);
        gpu_err_check(cudaGetLastError());
        gpu_err_check(cudaMemsetAsync(d_read_count_,   0, sizeof(uint32_t)));
        gpu_err_check(cudaMemsetAsync(d_write_count_,  0, sizeof(uint32_t)));
        gpu_err_check(cudaMemsetAsync(d_insert_count_, 0, sizeof(uint32_t)));
        uint32_t op_blocks = (submit_dest.curr_num_ops + block_size - 1) / block_size;
        countReadWriteInsertKeys<<<op_blocks, block_size>>>(
            submit_dest.d_read_keys,
            submit_dest.d_write_keys,
            submit_dest.d_insert_keys,
            submit_dest.curr_num_ops,
            d_read_count_,
            d_write_count_,
            d_insert_count_);
        gpu_err_check(cudaGetLastError());
        k_copy_scalars_to_mapped<<<1, 1>>>(
            d_read_count_,   d_mapped_read_cnt_,
            d_write_count_,  d_mapped_write_cnt_,
            d_insert_count_, d_mapped_insert_cnt_);
        gpu_err_check(cudaStreamSynchronize(0));
        submit_dest.curr_num_read_ops   = h_mapped_[1];
        submit_dest.curr_num_write_ops  = h_mapped_[2];
        submit_dest.curr_num_insert_ops = h_mapped_[3];
    } else {
        submit_dest.curr_num_read_ops   = 0;
        submit_dest.curr_num_write_ops  = 0;
        submit_dest.curr_num_insert_ops = 0;
    }
}

} // namespace epic::ycsb
