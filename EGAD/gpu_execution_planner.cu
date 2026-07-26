//
// Created by Shujian Qian on 2023-08-13.
//

#include "gpu_execution_planner.h"

#include <type_traits>

#include <cub/cub.cuh>

#include "util_math.h"
#include "util_log.h"
#include "util_arch.h"
#include "util_gpu_error_check.cuh"
#include "util_cub_reverse_iterator.cuh"

// TODO: this is really bad, fix when I have time
#include <benchmarks/tpcc_gpu_txn.cuh>
#include <benchmarks/tpcc_txn.h>
#include <benchmarks/ycsb_txn.h>
#include <benchmarks/micro_txn.h>
#include <benchmarks/ycsb_gpu_txn.cuh>

namespace epic {

namespace {

size_t allocateDeviceArray(
    Allocator &allocator, void *&ptr, size_t size, std::string_view name, size_t &total_allocated_size)
{
    epic::Logger &logger = epic::Logger::GetInstance();
    size = AlignTo(size, kDeviceCacheLineSize);
    logger.Trace("Allocating {} bytes for {}", formatSizeBytes(size), name);
    ptr = allocator.Allocate(size);
    total_allocated_size += size;
    return size;
}

} // namespace

template <typename TxnExecPlanArrayType>
void GpuTableExecutionPlanner<TxnExecPlanArrayType>::Initialize()
{
    epic::Logger &logger = epic::Logger::GetInstance();
    size_t total_allocated_size = 0;

    logger.Info("Initializing GPU table {}", name);

    //    allocateDeviceArray(allocator, d_records, max_num_records * record_size, "records", total_allocated_size);
    //    allocateDeviceArray(allocator, d_temp_versions, max_num_ops * record_size, "temp versions",
    //    total_allocated_size);

    // Arrays for submitter and stager
    // first for uint32_t * d_all_keys = nullptr;
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_all_keys), max_num_ops * sizeof(uint32_t), "all keys",
                total_allocated_size);
    // then for uint32_t * d_write_keys = nullptr;
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_read_keys), max_num_ops * sizeof(uint32_t), "read keys", total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_write_keys), max_num_ops * sizeof(uint32_t),
                        "write keys", total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_insert_keys), max_num_ops * sizeof(uint32_t),
                        "insert keys", total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_op_is_insert), max_num_ops * sizeof(uint8_t),
                        "op is_insert flags", total_allocated_size);

    // End of arrays for submitter and stager

    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_num_ops), max_num_txns * sizeof(uint32_t), "num ops",
        total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_op_offsets), max_num_txns * sizeof(uint32_t),
        "op offsets", total_allocated_size);
    cudaMemset(d_op_offsets, 0, max_num_txns * sizeof(uint32_t));
    allocateDeviceArray(allocator, d_submitted_ops, max_num_ops * sizeof(op_t), "submitted ops", total_allocated_size);
    allocateDeviceArray(allocator, d_sorted_ops, max_num_ops * sizeof(op_t), "sorted ops", total_allocated_size);
    allocateDeviceArray(
        allocator, d_write_ops_before, max_num_ops * sizeof(uint32_t), "write ops before", total_allocated_size);
    allocateDeviceArray(
        allocator, d_write_ops_after, max_num_ops * sizeof(uint32_t), "write ops after", total_allocated_size);
    allocateDeviceArray(allocator, d_rw_ops_type, max_num_ops * sizeof(uint8_t), "rw ops type", total_allocated_size);
    allocateDeviceArray(allocator, d_tver_write_ops_before, max_num_ops * sizeof(uint32_t), "temp write ops before",
        total_allocated_size);
    allocateDeviceArray(
        allocator, d_rw_locations, max_num_ops * sizeof(uint32_t), "rw locations", total_allocated_size);
    allocateDeviceArray(
        allocator, d_copy_dep_locations, max_num_ops * sizeof(uint32_t), "copy dep locations", total_allocated_size);
    allocateDeviceArray(
        allocator, d_copy_dep_locations, max_num_txns * sizeof(uint32_t *), "txn base pointer", total_allocated_size);
    scratch_array_bytes = std::max(4 * max_num_ops * sizeof(op_t), 2048lu);
    allocateDeviceArray(allocator, d_scratch_array, scratch_array_bytes, "scratch array", total_allocated_size);

    logger.Info("Total allocated size for {}: {}", name, formatSizeBytes(total_allocated_size));

    cudaStream_t stream;
    gpu_err_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_stream = stream;
}

template <typename TxnExecPlanArrayType>
void GpuTableExecutionPlanner<TxnExecPlanArrayType>::SubmitOps(
    CalcNumOpsFunc pre_submit_ops_func, SubmitOpsFunc submit_ops_func)
{}

namespace {
struct SameRow
{
    __host__ __device__ __forceinline__ bool operator()(op_t a, op_t b) const
    {
        return GET_RECORD_ID(a) == GET_RECORD_ID(b);
    }
};
struct WriteExtractor
{
    __host__ __device__ __forceinline__ uint32_t operator()(op_t op) const
    {
        return GET_R_W(op) == write_op;
    }
};
struct VersionWriteExtractor
{
    __host__ __device__ __forceinline__ uint32_t operator()(OperationT op) const
    {
        return op == OperationT::VERSION_WRITE;
    }
};
template<typename OpInputIt, typename WBeforeInputIt, typename WAfterInputIt, typename OutputIt>
__global__ void calcOperationType(
    OpInputIt op_input_it, WBeforeInputIt w_before, WAfterInputIt w_after, OutputIt output, size_t size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    OperationT retval;
    if (GET_R_W(op_input_it[idx]) == write_op)
    {
        if (w_after[idx] == 0)
        {
            retval = OperationT::RECORD_B_WRITE;
        }
        else
        {
            retval = OperationT::VERSION_WRITE;
        }
    }
    else
    {
        if (w_before[idx] == 0)
        {
            retval = OperationT::RECORD_A_READ;
        }
        else
        {
            if (w_after[idx] == 0)
            {
                retval = OperationT::RECORD_B_READ;
            }
            else
            {
                retval = OperationT::VERSION_READ;
            }
        }
    }

    output[idx] = retval;
}

template <typename GpuTxnExecPlanArrayType>
__global__ void scatterRWLocation(op_t *sorted_ops, OperationT *op_types, uint32_t *ver_writes_before,
    GpuTxnExecPlanArrayType exec_plan, uint32_t num_ops)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ops)
    {
        return;
    }
    uint32_t rw_location;
    switch (op_types[idx])
    {
    case OperationT::RECORD_A_READ:
    case OperationT::RECORD_A_WRITE:
        rw_location = loc_record_a;
        break;
    case OperationT::RECORD_B_READ:
    case OperationT::RECORD_B_WRITE:
        rw_location = loc_record_b;
        break;
    case OperationT::VERSION_READ:
        rw_location = ver_writes_before[idx] - 1;
        break;
    case OperationT::VERSION_WRITE:
        rw_location = ver_writes_before[idx];
        break;
    default:
        assert(false);
    }
    uint32_t txn_id = GET_TXN_ID(sorted_ops[idx]);
    uint32_t offset = GET_OFFSET(sorted_ops[idx]);
    uint32_t *txn_base_ptr = reinterpret_cast<uint32_t *>(exec_plan.getTxn(txn_id)->data);
    txn_base_ptr[offset] = rw_location;
}
} // namespace


/*
This is pretty much cuda code for executing all the steps written in the paper to get an execution plan for a table.
*/
template <typename TxnExecPlanArrayType>
void GpuTableExecutionPlanner<TxnExecPlanArrayType>::InitializeExecutionPlan()
{
    epic::Logger &logger = epic::Logger::GetInstance();
    logger.Info("Initializing execution plan for GPU table {}", name);

    if (curr_num_ops == 0)
    {
        return;
    }


    gpu_err_check(cub::DeviceRadixSort::SortKeys(d_scratch_array, scratch_array_bytes,
        static_cast<op_t *>(d_submitted_ops), static_cast<op_t *>(d_sorted_ops), curr_num_ops, 32, sizeof(op_t) * 8,
        std::any_cast<cudaStream_t>(cuda_stream)));


    using IsWriteIter = cub::TransformInputIterator<uint32_t, WriteExtractor, op_t *>;
    IsWriteIter w_before_val(static_cast<op_t *>(d_sorted_ops), WriteExtractor());
    gpu_err_check(cub::DeviceScan::ExclusiveSumByKey(d_scratch_array, scratch_array_bytes,
        static_cast<op_t *>(d_sorted_ops), w_before_val, static_cast<uint32_t *>(d_write_ops_before), curr_num_ops,
        SameRow(), std::any_cast<cudaStream_t>(cuda_stream)));


    IsWriteIter w_after_val(static_cast<op_t *>(d_sorted_ops) + curr_num_ops - 1, WriteExtractor());
    ReverseIterator<uint32_t, IsWriteIter> rev_write_after_val(w_after_val);
    ReverseIterator<op_t, op_t *> rev_sorted_ops(static_cast<op_t *>(d_sorted_ops) + curr_num_ops - 1);
    ReverseIterator<uint32_t, uint32_t *> rev_write_ops_after(
        static_cast<uint32_t *>(d_write_ops_after) + curr_num_ops - 1);
    gpu_err_check(cub::DeviceScan::ExclusiveSumByKey(d_scratch_array, scratch_array_bytes, rev_sorted_ops,
        rev_write_after_val, rev_write_ops_after, curr_num_ops, SameRow(), std::any_cast<cudaStream_t>(cuda_stream)));

    calcOperationType<<<(curr_num_ops + 255) / 256, 256, 0, std::any_cast<cudaStream_t>(cuda_stream)>>>(
        static_cast<op_t *>(d_sorted_ops), static_cast<uint32_t *>(d_write_ops_before),
        static_cast<uint32_t *>(d_write_ops_after), static_cast<OperationT *>(d_rw_ops_type), curr_num_ops);

    gpu_err_check(cudaGetLastError());

    using IsVersionWriteIter = cub::TransformInputIterator<uint32_t, VersionWriteExtractor, OperationT *>;
    IsVersionWriteIter version_write_val(static_cast<OperationT *>(d_rw_ops_type), VersionWriteExtractor());
    gpu_err_check(cub::DeviceScan::ExclusiveSum(d_scratch_array, scratch_array_bytes, version_write_val,
        static_cast<uint32_t *>(d_tver_write_ops_before), curr_num_ops, std::any_cast<cudaStream_t>(cuda_stream)));

    // TODO: fix this disgusting stuff by moving everything into a .cuh file
    using GpuTxnArrayType =
        typename std::conditional_t<std::is_same_v<TxnExecPlanArrayType, tpcc::TpccTxnExecPlanArrayT>,
            tpcc::TpccGpuTxnArrayT,
            typename std::conditional_t<std::is_same_v<TxnExecPlanArrayType, TxnArray<ycsb::YcsbExecPlan>>,
                ycsb::YcsbGpuTxnArrayT, GpuTxnArray>>;
    scatterRWLocation<<<(curr_num_ops + 255) / 256, 256, 0, std::any_cast<cudaStream_t>(cuda_stream)>>>(
        static_cast<op_t *>(d_sorted_ops), static_cast<OperationT *>(d_rw_ops_type),
        static_cast<uint32_t *>(d_tver_write_ops_before), GpuTxnArrayType(exec_plan),
        curr_num_ops);
    // TODO: again, disgusting hack
    static_assert(
        std::is_same_v<TxnExecPlanArrayType, tpcc::TpccTxnExecPlanArrayT> || std::is_same_v < TxnExecPlanArrayType,
        TxnArray<ycsb::YcsbExecPlan>> || std::is_same_v < TxnExecPlanArrayType, TxnArray<micro::MicroTxnExecPlan>>);
    gpu_err_check(cudaGetLastError());
}

template <typename TxnExecPlanArrayType>
void GpuTableExecutionPlanner<TxnExecPlanArrayType>::FinishInitialization()
{
    if (curr_num_ops == 0)
    {
        return;
    }

    auto &logger = Logger::GetInstance();
    gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(cuda_stream)));

}

template <typename TxnExecPlanArrayType>
void GpuTableExecutionPlanner<TxnExecPlanArrayType>::ScatterOpLocations() {}

template class GpuTableExecutionPlanner<tpcc::TpccTxnExecPlanArrayT>;
template class GpuTableExecutionPlanner<TxnArray<ycsb::YcsbExecPlan>>;
template class GpuTableExecutionPlanner<TxnArray<micro::MicroTxnExecPlan>>;

} // namespace epic