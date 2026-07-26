//
// Created by Shujian Qian on 2023-10-11.
//

#include "benchmarks/tpcc_gpu_submitter.h"
#include "execution_planner.h"
#include "tpcc_gpu_txn.cuh"

#include "util_log.h"
#include "util_gpu_error_check.cuh"

#include <thrust/for_each.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/system/cuda/detail/reduce.h>

namespace epic::tpcc {

template <typename TxnParamArrayType>
TpccGpuSubmitter<TxnParamArrayType>::TpccGpuSubmitter(TableSubmitDest warehouse_submit_dest, TableSubmitDest district_submit_dest,
    TableSubmitDest customer_submit_dest, TableSubmitDest history_submit_dest, TableSubmitDest new_order_submit_dest,
    TableSubmitDest order_submit_dest, TableSubmitDest order_line_submit_dest, TableSubmitDest item_submit_dest,
    TableSubmitDest stock_submit_dest)
    : TpccSubmitter<TxnParamArrayType>(warehouse_submit_dest, district_submit_dest, customer_submit_dest, history_submit_dest,
          new_order_submit_dest, order_submit_dest, order_line_submit_dest, item_submit_dest, stock_submit_dest)
{
    for (int i = 0; i < 9; i++)
    {
        cudaStream_t stream;
        gpu_err_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        // gpu_err_check(cudaStreamCreate(&stream));
        cuda_streams.emplace_back(stream);
    }
    // Allocate per-table derive-pass scratch: device counters + mapped host
    // readbacks. 9 tables × 3 counters (read/write/insert) = 27 scalars.
    constexpr size_t kBytes = kNumTablesForCounters * kCountersPerTable * sizeof(uint32_t);
    gpu_err_check(cudaMalloc(&d_counters_, kBytes));
    gpu_err_check(cudaHostAlloc(&h_counters_mapped, kBytes, cudaHostAllocMapped));
    memset(h_counters_mapped, 0, kBytes);
    gpu_err_check(cudaHostGetDevicePointer(&d_counters_mapped, h_counters_mapped, 0));
}

template <typename TxnParamArrayType>
TpccGpuSubmitter<TxnParamArrayType>::~TpccGpuSubmitter()
{
    for (auto &stream : cuda_streams)
    {
        if (stream.has_value())
        {
            gpu_err_check(cudaStreamDestroy(std::any_cast<cudaStream_t>(stream)));
        }
    }
    if (d_counters_)        { cudaFree(d_counters_);             d_counters_ = nullptr; }
    if (h_counters_mapped)  { cudaFreeHost(h_counters_mapped);   h_counters_mapped = nullptr; }
}

struct TpccNumOps
{
    uint32_t *warehouse_num_ops;
    uint32_t *district_num_ops;
    uint32_t *customer_num_ops;
    uint32_t *history_num_ops;
    uint32_t *order_num_ops;
    uint32_t *new_order_num_ops;
    uint32_t *order_line_num_ops;
    uint32_t *item_num_ops;
    uint32_t *stock_num_ops;
};

struct TpccSubmitLocations
{
    uint32_t *warehouse_offset;
    uint32_t *district_offset;
    uint32_t *customer_offset;
    uint32_t *history_offset;
    uint32_t *order_offset;
    uint32_t *new_order_offset;
    uint32_t *order_line_offset;
    uint32_t *item_offset;
    uint32_t *stock_offset;
    void *warehouse_dest;
    void *district_dest;
    void *customer_dest;
    void *history_dest;
    void *order_dest;
    void *new_order_dest;
    void *order_line_dest;
    void *item_dest;
    void *stock_dest;
    // Per-table parallel byte-array marking which submitted ops are
    // insert-emitting (1) vs not (0). Only the NewOrder transaction emits
    // inserts (Order / NewOrder / OrderLine writes); every other op is 0.
    uint8_t *warehouse_is_insert;
    uint8_t *district_is_insert;
    uint8_t *customer_is_insert;
    uint8_t *history_is_insert;
    uint8_t *order_is_insert;
    uint8_t *new_order_is_insert;
    uint8_t *order_line_is_insert;
    uint8_t *item_is_insert;
    uint8_t *stock_is_insert;
};

// Helper kernels for the per-table derive pass that runs after submitTpccTxn
// completes. These mirror ycsb_gpu_submitter.cu's deriveStagerKeys /
// countReadWriteInsertKeys pattern, replicated per table because each TPC-C
// table has its own ops/op_is_insert arrays and its own d_*_keys outputs.
__global__ void k_copy_one_scalar_to_mapped(const uint32_t* src, uint32_t* dst)
{
    if (threadIdx.x == 0 && src && dst) *dst = *src;
}

__global__ void deriveStagerKeysTpcc(const op_t* ops, const uint8_t* op_is_insert, uint32_t n,
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

__global__ void countReadWriteInsertKeysTpcc(const uint32_t* read_keys,
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

    if (read_keys[tid]   != SENT) atomicAdd(d_read_count,   1u);
    if (write_keys[tid]  != SENT) atomicAdd(d_write_count,  1u);
    if (insert_keys[tid] != SENT) atomicAdd(d_insert_count, 1u);
}

static __device__ __forceinline__ void prepareSubmitTpccTxn(
    int txn_id, NewOrderTxnParams<FixedSizeTxn> *txn, TpccNumOps num_ops)
{
    num_ops.warehouse_num_ops[txn_id] = 1;
    num_ops.district_num_ops[txn_id] = 2;
    num_ops.customer_num_ops[txn_id] = 1;
    num_ops.history_num_ops[txn_id] = 0;
    num_ops.order_num_ops[txn_id] = 1;
    num_ops.new_order_num_ops[txn_id] = 1;
    uint32_t num_items = txn->num_items;
    num_ops.order_line_num_ops[txn_id] = num_items;
    num_ops.item_num_ops[txn_id] = num_items;
    num_ops.stock_num_ops[txn_id] = num_items * 2;
}

static __device__ __forceinline__ void prepareSubmitTpccTxn(int txn_id, PaymentTxnParams *txn, TpccNumOps num_ops)
{
    num_ops.warehouse_num_ops[txn_id] = 2;
    num_ops.district_num_ops[txn_id] = 2;
    num_ops.customer_num_ops[txn_id] = 2;
    num_ops.history_num_ops[txn_id] = 0; /* TODO: deal with history table later */
    num_ops.order_num_ops[txn_id] = 0;
    num_ops.new_order_num_ops[txn_id] = 0;
    num_ops.order_line_num_ops[txn_id] = 0;
    num_ops.item_num_ops[txn_id] = 0;
    num_ops.stock_num_ops[txn_id] = 0;
}

static __device__ __forceinline__ void prepareSubmitTpccTxn(int txn_id, OrderStatusTxnParams *txn, TpccNumOps num_ops)
{
    num_ops.warehouse_num_ops[txn_id] = 0;
    num_ops.district_num_ops[txn_id] = 0;
    num_ops.customer_num_ops[txn_id] = 1;
    num_ops.history_num_ops[txn_id] = 0;
    num_ops.order_num_ops[txn_id] = 1;
    num_ops.new_order_num_ops[txn_id] = 0;
    num_ops.order_line_num_ops[txn_id] = txn->num_items;
    num_ops.item_num_ops[txn_id] = 0;
    num_ops.stock_num_ops[txn_id] = 0;
}

static void __device__ __forceinline__ prepareSubmitTpccTxn(int txn_id, DeliveryTxnParams *txn, TpccNumOps num_ops)
{
    int orderline_ops = 0;
    for (int i = 0; i < 10; ++i)
    {
        orderline_ops += txn->num_items[i];
    }
    // printf("txn[%d] submit total_items[%d] ptr[%p] "
    //     "item0[%d] "
    //     "item1[%d] "
    //     "item2[%d] "
    //     "item3[%d] "
    //     "item4[%d] "
    //     "item5[%d] "
    //     "item6[%d] "
    //     "item7[%d] "
    //     "item8[%d] "
    //     "item9[%d] "
    //     "\n", txn_id, orderline_ops, txn,
    //     txn->num_items[0],
    //     txn->num_items[1],
    //     txn->num_items[2],
    //     txn->num_items[3],
    //     txn->num_items[4],
    //     txn->num_items[5],
    //     txn->num_items[6],
    //     txn->num_items[7],
    //     txn->num_items[8],
    //     txn->num_items[9]
    //     );
    //
    num_ops.warehouse_num_ops[txn_id] = 0;
    num_ops.district_num_ops[txn_id] = 0;
    num_ops.customer_num_ops[txn_id] = 20;
    num_ops.history_num_ops[txn_id] = 0;
    num_ops.order_num_ops[txn_id] = 20;
    num_ops.new_order_num_ops[txn_id] = 10;
    num_ops.order_line_num_ops[txn_id] = orderline_ops * 2;
    num_ops.item_num_ops[txn_id] = 0;
    num_ops.stock_num_ops[txn_id] = 0;

    // num_ops.new_order_num_ops[txn_id] = 0;
    // num_ops.customer_num_ops[txn_id] = 0;
    // num_ops.order_num_ops[txn_id] = 0; // TODO: remove
    // num_ops.order_line_num_ops[txn_id] = 0; // TODO: remove
    // num_ops.order_line_num_ops[txn_id] = 0; // TODO: remove
}

static void __device__ __forceinline__ prepareSubmitTpccTxn(int txn_id, StockLevelTxnParams *txn, TpccNumOps num_ops)
{
    num_ops.warehouse_num_ops[txn_id] = 0;
    num_ops.district_num_ops[txn_id] = 0;
    num_ops.customer_num_ops[txn_id] = 0;
    num_ops.history_num_ops[txn_id] = 0;
    num_ops.order_num_ops[txn_id] = 0;
    num_ops.new_order_num_ops[txn_id] = 0;
    num_ops.order_line_num_ops[txn_id] = 0;
    num_ops.item_num_ops[txn_id] = 0;
    num_ops.stock_num_ops[txn_id] = txn->num_items;
}

template <typename GpuTxnArrayType>
static __global__ void prepareSubmitTpccTxn(GpuTxnArrayType txn_array, TpccNumOps num_ops)
{
    int txn_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (txn_id >= txn_array.num_txns)
    {
        return;
    }
    BaseTxn *base_txn = txn_array.getTxn(txn_id);
    switch (static_cast<TpccTxnType>(base_txn->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        prepareSubmitTpccTxn(txn_id, reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(base_txn->data), num_ops);
        break;
    case TpccTxnType::PAYMENT:
        prepareSubmitTpccTxn(txn_id, reinterpret_cast<PaymentTxnParams *>(base_txn->data), num_ops);
        break;
    case TpccTxnType::ORDER_STATUS:
        prepareSubmitTpccTxn(txn_id, reinterpret_cast<OrderStatusTxnParams *>(base_txn->data), num_ops);
        break;
    case TpccTxnType::DELIVERY:
        prepareSubmitTpccTxn(txn_id, reinterpret_cast<DeliveryTxnParams *>(base_txn->data), num_ops);
        break;
    case TpccTxnType::STOCK_LEVEL:
        prepareSubmitTpccTxn(txn_id, reinterpret_cast<StockLevelTxnParams *>(base_txn->data), num_ops);
        break;
    default:
        assert(false);
    }
}

static __device__ __forceinline__ void submitTpccTxn(
    int txn_id, NewOrderTxnParams<FixedSizeTxn> *txn, TpccSubmitLocations submit_loc)
{
    static_cast<uint64_t *>(submit_loc.warehouse_dest)[submit_loc.warehouse_offset[txn_id]] = CREATE_OP(
        txn->warehouse_id, txn_id, read_op, offsetof(NewOrderExecPlan<FixedSizeTxn>, warehouse_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.district_dest)[submit_loc.district_offset[txn_id]] = CREATE_OP(
        txn->district_id, txn_id, read_op, offsetof(NewOrderExecPlan<FixedSizeTxn>, district_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.district_dest)[submit_loc.district_offset[txn_id] + 1] = CREATE_OP(
        txn->district_id, txn_id, write_op, offsetof(NewOrderExecPlan<FixedSizeTxn>, district_write_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.customer_dest)[submit_loc.customer_offset[txn_id]] = CREATE_OP(
        txn->customer_id, txn_id, read_op, offsetof(NewOrderExecPlan<FixedSizeTxn>, customer_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.order_dest)[submit_loc.order_offset[txn_id]] = CREATE_OP(
        txn->order_id, txn_id, write_op, offsetof(NewOrderExecPlan<FixedSizeTxn>, order_loc) / sizeof(uint32_t));
    // INSERT marker: the Order write above creates a new row.
    submit_loc.order_is_insert[submit_loc.order_offset[txn_id]] = 1;

    static_cast<uint64_t *>(submit_loc.new_order_dest)[submit_loc.new_order_offset[txn_id]] =
        CREATE_OP(txn->new_order_id, txn_id, write_op,
            offsetof(NewOrderExecPlan<FixedSizeTxn>, new_order_loc) / sizeof(uint32_t));
    // INSERT marker: the NewOrder write above creates a new row.
    submit_loc.new_order_is_insert[submit_loc.new_order_offset[txn_id]] = 1;

    for (int i = 0; i < txn->num_items; i++)
    {
        static_cast<uint64_t *>(submit_loc.order_line_dest)[submit_loc.order_line_offset[txn_id] + i] =
            CREATE_OP(txn->items[i].order_line_id, txn_id, write_op,
                offsetof(NewOrderExecPlan<FixedSizeTxn>, item_plans[i].orderline_loc) / sizeof(uint32_t));
        // INSERT marker: each OrderLine write creates a new row.
        submit_loc.order_line_is_insert[submit_loc.order_line_offset[txn_id] + i] = 1;

        static_cast<uint64_t *>(submit_loc.item_dest)[submit_loc.item_offset[txn_id] + i] =
            CREATE_OP(txn->items[i].item_id, txn_id, read_op,
                offsetof(NewOrderExecPlan<FixedSizeTxn>, item_plans[i].item_loc) / sizeof(uint32_t));
        static_cast<uint64_t *>(submit_loc.stock_dest)[submit_loc.stock_offset[txn_id] + i * 2] =
            CREATE_OP(txn->items[i].stock_id, txn_id, read_op,
                offsetof(NewOrderExecPlan<FixedSizeTxn>, item_plans[i].stock_read_loc) / sizeof(uint32_t));
        static_cast<uint64_t *>(submit_loc.stock_dest)[submit_loc.stock_offset[txn_id] + i * 2 + 1] =
            CREATE_OP(txn->items[i].stock_id, txn_id, write_op,
                offsetof(NewOrderExecPlan<FixedSizeTxn>, item_plans[i].stock_write_loc) / sizeof(uint32_t));
    }
    __threadfence(); /* TODO: remove this */
}

static __device__ __forceinline__ void submitTpccTxn(int txn_id, PaymentTxnParams *txn, TpccSubmitLocations submit_loc)
{
    static_cast<uint64_t *>(submit_loc.warehouse_dest)[submit_loc.warehouse_offset[txn_id]] = CREATE_OP(
        txn->warehouse_id, txn_id, read_op, offsetof(PaymentTxnExecPlan, warehouse_read_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.warehouse_dest)[submit_loc.warehouse_offset[txn_id] + 1] = CREATE_OP(
        txn->warehouse_id, txn_id, write_op, offsetof(PaymentTxnExecPlan, warehouse_write_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.district_dest)[submit_loc.district_offset[txn_id]] = CREATE_OP(
        txn->district_id, txn_id, read_op, offsetof(PaymentTxnExecPlan, district_read_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.district_dest)[submit_loc.district_offset[txn_id] + 1] = CREATE_OP(
        txn->district_id, txn_id, write_op, offsetof(PaymentTxnExecPlan, district_write_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.customer_dest)[submit_loc.customer_offset[txn_id]] = CREATE_OP(
        txn->customer_id, txn_id, read_op, offsetof(PaymentTxnExecPlan, customer_read_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.customer_dest)[submit_loc.customer_offset[txn_id] + 1] = CREATE_OP(
        txn->customer_id, txn_id, write_op, offsetof(PaymentTxnExecPlan, customer_write_loc) / sizeof(uint32_t));
    __threadfence(); /* TODO: remove this */
}

static __device__ __forceinline__ void submitTpccTxn(int txn_id, OrderStatusTxnParams *txn, TpccSubmitLocations submit_loc)
{
    static_cast<uint64_t *>(submit_loc.customer_dest)[submit_loc.customer_offset[txn_id]] = CREATE_OP(
        txn->customer_id, txn_id, read_op, offsetof(OrderStatusTxnExecPlan, customer_loc) / sizeof(uint32_t));
    static_cast<uint64_t *>(submit_loc.order_dest)[submit_loc.order_offset[txn_id]] = CREATE_OP(
        txn->order_id, txn_id, read_op, offsetof(OrderStatusTxnExecPlan, order_loc) / sizeof(uint32_t));
    for (int i = 0; i < txn->num_items; i++)
    {
        static_cast<uint64_t *>(submit_loc.order_line_dest)[submit_loc.order_line_offset[txn_id] + i] =
            CREATE_OP(txn->orderline_ids[i], txn_id, read_op,
                offsetof(OrderStatusTxnExecPlan, orderline_locs[i]) / sizeof(uint32_t));
    }
}

static __device__ __forceinline__ void submitTpccTxn(int txn_id, DeliveryTxnParams *txn, TpccSubmitLocations submit_loc)
{
    int orderline_ops = 0;
    for (int i = 0; i < 10; ++i)
    {
        int loc_offset = i * 2;
        static_cast<uint64_t *>(submit_loc.customer_dest)[submit_loc.customer_offset[txn_id] + loc_offset] = CREATE_OP(
            txn->customer_id[i], txn_id, read_op, offsetof(DeliveryTxnExecPlan, customer_read_locs[i]) / sizeof(uint32_t));
        static_cast<uint64_t *>(submit_loc.customer_dest)[submit_loc.customer_offset[txn_id] + loc_offset + 1] = CREATE_OP(
            txn->customer_id[i], txn_id, write_op, offsetof(DeliveryTxnExecPlan, customer_write_locs[i]) / sizeof(uint32_t));

        static_cast<uint64_t *>(submit_loc.new_order_dest)[submit_loc.new_order_offset[txn_id] + i] = CREATE_OP(
            txn->new_order_id[i], txn_id, read_op, offsetof(DeliveryTxnExecPlan, new_order_read_locs[i]) / sizeof(uint32_t));

        static_cast<uint64_t *>(submit_loc.order_dest)[submit_loc.order_offset[txn_id] + loc_offset] = CREATE_OP(
            txn->order_id[i], txn_id, read_op, offsetof(DeliveryTxnExecPlan, order_read_locs[i]) / sizeof(uint32_t));
        static_cast<uint64_t *>(submit_loc.order_dest)[submit_loc.order_offset[txn_id] + loc_offset + 1] = CREATE_OP(
            txn->order_id[i], txn_id, write_op, offsetof(DeliveryTxnExecPlan, order_write_locs[i]) / sizeof(uint32_t));


        for (int j = 0; j < txn->num_items[i]; ++j)
        {
            static_cast<uint64_t *>(submit_loc.order_line_dest)[submit_loc.order_line_offset[txn_id] + orderline_ops] =
                CREATE_OP(txn->orderline_ids[i][j], txn_id, read_op,
                    offsetof(DeliveryTxnExecPlan, orderline_read_locs[i][j]) / sizeof(uint32_t));
            orderline_ops++;
            static_cast<uint64_t *>(submit_loc.order_line_dest)[submit_loc.order_line_offset[txn_id] + orderline_ops] =
                CREATE_OP(txn->orderline_ids[i][j], txn_id, write_op,
                    offsetof(DeliveryTxnExecPlan, orderline_write_locs[i][j]) / sizeof(uint32_t));
            orderline_ops++;
        }
    }
}

static __device__ __forceinline__ void submitTpccTxn(
    int txn_id, StockLevelTxnParams *txn, TpccSubmitLocations submit_loc)
{
    uint32_t stock_offset = submit_loc.stock_offset[txn_id];
    for (uint32_t i = 0; i < txn->num_items; ++i)
    {
        static_cast<uint64_t *>(submit_loc.stock_dest)[stock_offset + i] = CREATE_OP(
            txn->stock_ids[i], txn_id, read_op, offsetof(StockLevelTxnExecPlan, stock_read_locs[i]) / sizeof(uint32_t));
    }
}

template <typename GpuTxnArrayType>
static __global__ void submitTpccTxn(GpuTxnArrayType txn_array, TpccSubmitLocations submit_loc)
{
    int txn_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (txn_id >= txn_array.num_txns)
    {
        return;
    }
    BaseTxn *base_txn = txn_array.getTxn(txn_id);
    switch (static_cast<TpccTxnType>(base_txn->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        submitTpccTxn(txn_id, reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(base_txn->data), submit_loc);
        break;
    case TpccTxnType::PAYMENT:
        submitTpccTxn(txn_id, reinterpret_cast<PaymentTxnParams *>(base_txn->data), submit_loc);
        break;
    case TpccTxnType::ORDER_STATUS:
        submitTpccTxn(txn_id, reinterpret_cast<OrderStatusTxnParams *>(base_txn->data), submit_loc);
        break;
    case TpccTxnType::DELIVERY:
        submitTpccTxn(txn_id, reinterpret_cast<DeliveryTxnParams *>(base_txn->data), submit_loc);
        break;
    case TpccTxnType::STOCK_LEVEL:
        submitTpccTxn(txn_id, reinterpret_cast<StockLevelTxnParams *>(base_txn->data), submit_loc);
        break;
    default:
        assert(false);
    }
}

// Count each transaction's per-table operations, inclusive-scan the counts
// into per-txn offsets, then emit the packed op entries at those offsets.
template <typename TxnParamArrayType>
void TpccGpuSubmitter<TxnParamArrayType>::submit(TxnParamArrayType &txn_array)
{
    auto &logger = Logger::GetInstance();
    TpccNumOps num_ops = {.warehouse_num_ops = warehouse_submit_dest.d_num_ops,
        .district_num_ops = district_submit_dest.d_num_ops,
        .customer_num_ops = customer_submit_dest.d_num_ops,
        .history_num_ops = history_submit_dest.d_num_ops,
        .order_num_ops = order_submit_dest.d_num_ops,
        .new_order_num_ops = new_order_submit_dest.d_num_ops,
        .order_line_num_ops = order_line_submit_dest.d_num_ops,
        .item_num_ops = item_submit_dest.d_num_ops,
        .stock_num_ops = stock_submit_dest.d_num_ops};
    // launch a kernel to prepare the transactions
    prepareSubmitTpccTxn<<<(txn_array.num_txns + 1024) / 1024, 1024, 0, std::any_cast<cudaStream_t>(cuda_streams[0])>>>(
        TpccGpuTxnArrayT(txn_array), num_ops);

    gpu_err_check(cudaGetLastError());
    gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(cuda_streams[0])));
    /*
    This function performs an inclusive prefix sum (scan) on the number of operations for each transaction in a given TPCC table. 
    The result assigns memory offsets for each transaction's operations.
    */
    // perform inclusive scan to calculate the offset for each transaction
    // Instead of assigning each operation dynamically, this allows for a single, contiguous memory allocation.
    gpu_err_check(cub::DeviceScan::InclusiveSum(warehouse_submit_dest.temp_storage,
        warehouse_submit_dest.temp_storage_bytes, warehouse_submit_dest.d_num_ops,
        warehouse_submit_dest.d_op_offsets + 1, txn_array.num_txns, std::any_cast<cudaStream_t>(cuda_streams[0])));
    gpu_err_check(cub::DeviceScan::InclusiveSum(district_submit_dest.temp_storage,
        district_submit_dest.temp_storage_bytes, district_submit_dest.d_num_ops, district_submit_dest.d_op_offsets + 1,
        txn_array.num_txns, std::any_cast<cudaStream_t>(cuda_streams[1])));
    gpu_err_check(cub::DeviceScan::InclusiveSum(customer_submit_dest.temp_storage,
        customer_submit_dest.temp_storage_bytes, customer_submit_dest.d_num_ops, customer_submit_dest.d_op_offsets + 1,
        txn_array.num_txns, std::any_cast<cudaStream_t>(cuda_streams[2])));
    gpu_err_check(cub::DeviceScan::InclusiveSum(history_submit_dest.temp_storage,
        history_submit_dest.temp_storage_bytes, history_submit_dest.d_num_ops, history_submit_dest.d_op_offsets + 1,
        txn_array.num_txns, std::any_cast<cudaStream_t>(cuda_streams[3])));
    gpu_err_check(cub::DeviceScan::InclusiveSum(order_submit_dest.temp_storage, order_submit_dest.temp_storage_bytes,
        order_submit_dest.d_num_ops, order_submit_dest.d_op_offsets + 1, txn_array.num_txns,
        std::any_cast<cudaStream_t>(cuda_streams[4])));
    gpu_err_check(cub::DeviceScan::InclusiveSum(new_order_submit_dest.temp_storage,
        new_order_submit_dest.temp_storage_bytes, new_order_submit_dest.d_num_ops,
        new_order_submit_dest.d_op_offsets + 1, txn_array.num_txns, std::any_cast<cudaStream_t>(cuda_streams[5])));
    gpu_err_check(cub::DeviceScan::InclusiveSum(order_line_submit_dest.temp_storage,
        order_line_submit_dest.temp_storage_bytes, order_line_submit_dest.d_num_ops,
        order_line_submit_dest.d_op_offsets + 1, txn_array.num_txns, std::any_cast<cudaStream_t>(cuda_streams[6])));
    gpu_err_check(cub::DeviceScan::InclusiveSum(item_submit_dest.temp_storage, item_submit_dest.temp_storage_bytes,
        item_submit_dest.d_num_ops, item_submit_dest.d_op_offsets + 1, txn_array.num_txns,
        std::any_cast<cudaStream_t>(cuda_streams[7])));
    gpu_err_check(cub::DeviceScan::InclusiveSum(stock_submit_dest.temp_storage, stock_submit_dest.temp_storage_bytes,
        stock_submit_dest.d_num_ops, stock_submit_dest.d_op_offsets + 1, txn_array.num_txns,
        std::any_cast<cudaStream_t>(cuda_streams[8])));
    // copy the number of operations for each table to the host
    // These asynchronously copy the final offset value from the GPU to the CPU
    gpu_err_check(
        cudaMemcpyAsync(&warehouse_submit_dest.curr_num_ops, warehouse_submit_dest.d_op_offsets + txn_array.num_txns,
            sizeof(uint32_t), cudaMemcpyDeviceToHost, std::any_cast<cudaStream_t>(cuda_streams[0])));
    gpu_err_check(
        cudaMemcpyAsync(&district_submit_dest.curr_num_ops, district_submit_dest.d_op_offsets + txn_array.num_txns,
            sizeof(uint32_t), cudaMemcpyDeviceToHost, std::any_cast<cudaStream_t>(cuda_streams[1])));
    gpu_err_check(
        cudaMemcpyAsync(&customer_submit_dest.curr_num_ops, customer_submit_dest.d_op_offsets + txn_array.num_txns,
            sizeof(uint32_t), cudaMemcpyDeviceToHost, std::any_cast<cudaStream_t>(cuda_streams[2])));
    gpu_err_check(
        cudaMemcpyAsync(&history_submit_dest.curr_num_ops, history_submit_dest.d_op_offsets + txn_array.num_txns,
            sizeof(uint32_t), cudaMemcpyDeviceToHost, std::any_cast<cudaStream_t>(cuda_streams[3])));
    gpu_err_check(cudaMemcpyAsync(&order_submit_dest.curr_num_ops, order_submit_dest.d_op_offsets + txn_array.num_txns,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, std::any_cast<cudaStream_t>(cuda_streams[4])));
    gpu_err_check(
        cudaMemcpyAsync(&new_order_submit_dest.curr_num_ops, new_order_submit_dest.d_op_offsets + txn_array.num_txns,
            sizeof(uint32_t), cudaMemcpyDeviceToHost, std::any_cast<cudaStream_t>(cuda_streams[5])));
    gpu_err_check(
        cudaMemcpyAsync(&order_line_submit_dest.curr_num_ops, order_line_submit_dest.d_op_offsets + txn_array.num_txns,
            sizeof(uint32_t), cudaMemcpyDeviceToHost, std::any_cast<cudaStream_t>(cuda_streams[6])));
    gpu_err_check(cudaMemcpyAsync(&item_submit_dest.curr_num_ops, item_submit_dest.d_op_offsets + txn_array.num_txns,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, std::any_cast<cudaStream_t>(cuda_streams[7])));
    gpu_err_check(cudaMemcpyAsync(&stock_submit_dest.curr_num_ops, stock_submit_dest.d_op_offsets + txn_array.num_txns,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, std::any_cast<cudaStream_t>(cuda_streams[8])));

    // Set transaction submission locations
    TpccSubmitLocations locs = {
        .warehouse_offset = warehouse_submit_dest.d_op_offsets,
        .district_offset = district_submit_dest.d_op_offsets,
        .customer_offset = customer_submit_dest.d_op_offsets,
        .history_offset = history_submit_dest.d_op_offsets,
        .order_offset = order_submit_dest.d_op_offsets,
        .new_order_offset = new_order_submit_dest.d_op_offsets,
        .order_line_offset = order_line_submit_dest.d_op_offsets,
        .item_offset = item_submit_dest.d_op_offsets,
        .stock_offset = stock_submit_dest.d_op_offsets,
        .warehouse_dest = warehouse_submit_dest.d_submitted_ops,
        .district_dest = district_submit_dest.d_submitted_ops,
        .customer_dest = customer_submit_dest.d_submitted_ops,
        .history_dest = history_submit_dest.d_submitted_ops,
        .order_dest = order_submit_dest.d_submitted_ops,
        .new_order_dest = new_order_submit_dest.d_submitted_ops,
        .order_line_dest = order_line_submit_dest.d_submitted_ops,
        .item_dest = item_submit_dest.d_submitted_ops,
        .stock_dest = stock_submit_dest.d_submitted_ops,
        .warehouse_is_insert  = warehouse_submit_dest.d_op_is_insert,
        .district_is_insert   = district_submit_dest.d_op_is_insert,
        .customer_is_insert   = customer_submit_dest.d_op_is_insert,
        .history_is_insert    = history_submit_dest.d_op_is_insert,
        .order_is_insert      = order_submit_dest.d_op_is_insert,
        .new_order_is_insert  = new_order_submit_dest.d_op_is_insert,
        .order_line_is_insert = order_line_submit_dest.d_op_is_insert,
        .item_is_insert       = item_submit_dest.d_op_is_insert,
        .stock_is_insert      = stock_submit_dest.d_op_is_insert,
    };

    // Per-table memset of d_op_is_insert to 0 BEFORE submitTpccTxn writes
    // its per-op insert markers. Only NewOrder writes 1s (Order/NewOrder/
    // OrderLine emits — see the dispatch above); every other op leaves the
    // byte at 0. Memset is bounded by max_num_ops (the planner's allocation
    // size for d_op_is_insert), launched on the same stream as
    // submitTpccTxn so the kernel sees the zeros.
    auto memset_is_insert = [](TableSubmitDest& d, cudaStream_t s) {
        if (d.d_op_is_insert && d.max_num_ops > 0) {
            gpu_err_check(cudaMemsetAsync(d.d_op_is_insert, 0,
                d.max_num_ops * sizeof(uint8_t), s));
        }
    };
    cudaStream_t s0 = std::any_cast<cudaStream_t>(cuda_streams[0]);
    memset_is_insert(warehouse_submit_dest,  s0);
    memset_is_insert(district_submit_dest,   s0);
    memset_is_insert(customer_submit_dest,   s0);
    memset_is_insert(history_submit_dest,    s0);
    memset_is_insert(order_submit_dest,      s0);
    memset_is_insert(new_order_submit_dest,  s0);
    memset_is_insert(order_line_submit_dest, s0);
    memset_is_insert(item_submit_dest,       s0);
    memset_is_insert(stock_submit_dest,      s0);

    /*
    The submitTpccTxn function is responsible for scheduling and submitting TPCC transactions for execution on the GPU. 
    It takes transactions that have already been prepared (via prepareSubmitTpccTxn) and translates them into operation 
    commands that will be executed in the TPCC benchmark.

    It basically creates the operations for each transaction based on the transaction type and the prepared metadata, then 
    it writes it to the appropriate destination arrays for each table.
    */
    submitTpccTxn<<<(txn_array.num_txns + 1024) / 1024, 1024, 0, std::any_cast<cudaStream_t>(cuda_streams[0])>>>(
        TpccGpuTxnArrayT(txn_array), locs);

    gpu_err_check(cudaGetLastError());
    for (auto &stream : cuda_streams)
    {
        gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(stream)));
    }

    // Per-table derive pass: build d_all/read/write/insert_keys from the
    // canonical d_submitted_ops + d_op_is_insert, then count non-sentinel
    // entries per subset. Each table runs on its own stream so the 9 derive
    // passes overlap. The mapped-host counters are read after the final sync.
    {
        constexpr int B = 256;
        TableSubmitDest* dests[kNumTablesForCounters] = {
            &warehouse_submit_dest, &district_submit_dest, &customer_submit_dest,
            &history_submit_dest,   &order_submit_dest,    &new_order_submit_dest,
            &order_line_submit_dest, &item_submit_dest,    &stock_submit_dest
        };
        for (int t = 0; t < kNumTablesForCounters; ++t) {
            TableSubmitDest& d = *dests[t];
            cudaStream_t s = std::any_cast<cudaStream_t>(cuda_streams[t]);
            uint32_t n = d.curr_num_ops;
            uint32_t* dev_counters_t = d_counters_ + t * kCountersPerTable;
            uint32_t* mapped_counters_t = d_counters_mapped + t * kCountersPerTable;
            // Reset this table's 3 device counters; the count kernel atomic-
            // increments them. Always issue the memset so previous-epoch values
            // don't leak through when n shrinks.
            gpu_err_check(cudaMemsetAsync(dev_counters_t, 0, kCountersPerTable * sizeof(uint32_t), s));
            if (n > 0) {
                int G = (n + B - 1) / B;
                deriveStagerKeysTpcc<<<G, B, 0, s>>>(
                    reinterpret_cast<const op_t*>(d.d_submitted_ops),
                    d.d_op_is_insert, n,
                    d.d_all_keys, d.d_read_keys, d.d_write_keys, d.d_insert_keys);
                gpu_err_check(cudaPeekAtLastError());
                countReadWriteInsertKeysTpcc<<<G, B, 0, s>>>(
                    d.d_read_keys, d.d_write_keys, d.d_insert_keys, n,
                    dev_counters_t + 0, dev_counters_t + 1, dev_counters_t + 2);
                gpu_err_check(cudaPeekAtLastError());
            }
            // Copy this table's 3 counts to mapped host memory in a single
            // 1-thread kernel — bypasses the D2H copy engine.
            k_copy_one_scalar_to_mapped<<<1, 1, 0, s>>>(dev_counters_t + 0, mapped_counters_t + 0);
            k_copy_one_scalar_to_mapped<<<1, 1, 0, s>>>(dev_counters_t + 1, mapped_counters_t + 1);
            k_copy_one_scalar_to_mapped<<<1, 1, 0, s>>>(dev_counters_t + 2, mapped_counters_t + 2);
        }
        // Sync everything before reading the mapped counters back into the
        // SubmitDest references that the hybrid_staging path consumes.
        for (auto& stream : cuda_streams) {
            gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(stream)));
        }
        for (int t = 0; t < kNumTablesForCounters; ++t) {
            TableSubmitDest& d = *dests[t];
            uint32_t* mapped_counters_t = h_counters_mapped + t * kCountersPerTable;
            d.curr_num_read_ops   = mapped_counters_t[0];
            d.curr_num_write_ops  = mapped_counters_t[1];
            d.curr_num_insert_ops = mapped_counters_t[2];
        }
    }

    // log execution statistics
    logger.Info("num txns: {}", txn_array.num_txns);
    logger.Info("warehouse num ops: {}", warehouse_submit_dest.curr_num_ops);
    logger.Info("district num ops: {}", district_submit_dest.curr_num_ops);
    logger.Info("customer num ops: {}", customer_submit_dest.curr_num_ops);
    logger.Info("history num ops: {}", history_submit_dest.curr_num_ops);
    logger.Info("order num ops: {}", order_submit_dest.curr_num_ops);
    logger.Info("new order num ops: {}", new_order_submit_dest.curr_num_ops);
    logger.Info("order line num ops: {}", order_line_submit_dest.curr_num_ops);
    logger.Info("item num ops: {}", item_submit_dest.curr_num_ops);
    logger.Info("stock num ops: {}", stock_submit_dest.curr_num_ops);

}

template class TpccGpuSubmitter<TpccTxnParamArrayT>;

} // namespace epic::tpcc