//
// Created by Shujian Qian on 2023-10-25.
//
/*
This file contains the code necessary for executing TPCC transactions on the GPU, according to the specifications of TPCC.
*/

#include "tpcc_gpu_txn.cuh"

#include <benchmarks/tpcc_hybrid_executor.h>

#include <stdio.h>

#include <gpu_storage.cuh>
#include <util_gpu_error_check.cuh>
#include <util_arch.h>
#include <gpu_txn.cuh>
#include <util_warp_memory.cuh>
#include <util_gpu_transfer.h>
#include <util_log.h>

namespace epic::tpcc {

namespace {

constexpr uint32_t block_size = 128;
static_assert(block_size % kDeviceWarpSize == 0, "block_size must be a multiple of 32");
constexpr uint32_t num_warps = block_size / kDeviceWarpSize;

__device__ uint32_t txn_counter = 0; /* used for scheduling txns among threads */
const uint32_t zero = 0;

__device__ __forceinline__ void gpuExecTpccTxn(TpccRecords records, TpccVersions versions,
    NewOrderTxnParams<FixedSizeTxn> *params, NewOrderExecPlan<FixedSizeTxn> *plan, uint32_t epoch, uint32_t lane_id,
    uint32_t txn_id)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;
    constexpr uint32_t s_quantity_offset = offsetof(StockValue, s_quantity) / sizeof(uint32_t);
    constexpr uint32_t d_next_o_id_offset = offsetof(DistrictValue, d_next_o_id) / sizeof(uint32_t);


    uint32_t result = 0;
    gpuReadFromTableCoop(records.warehouse_record, versions.warehouse_version, params->warehouse_id,
        plan->warehouse_loc, epoch, result, lane_id);

    gpuReadFromTableCoop(records.district_record, versions.district_version, params->district_id, plan->district_loc,
        epoch, result, lane_id);
    if (lane_id == d_next_o_id_offset)
    {
        result = params->next_order_id;
    }
    gpuWriteToTableCoop(records.district_record, versions.district_version, params->district_id,
        plan->district_write_loc, epoch, result, lane_id, records.district_dirty_v1, records.district_dirty_v2);

    gpuReadFromTableCoop(records.customer_record, versions.customer_version, params->customer_id, plan->customer_loc,
        epoch, result, lane_id);

    // Populate o_c_id and o_ol_cnt in the Order record so
    // cpu_records.order_record carries the full Order data (standard TPCC
    // semantics). Without this, the executor writes the read of the customer
    // record straight into the order slot, leaving o_c_id and o_ol_cnt as
    // arbitrary bytes from the customer record — typically zero. The aux
    // index has this data but only on GPU, so it would not survive a crash
    // and recovery could not reconstruct it. Two extra
    // per-NewOrder GPU writes; ~0.5us/epoch at 45k NewOrders, well below
    // noise. The Order/NewOrder records' other slots are still filled with
    // customer-record bytes (harmless padding for fields we don't read via
    // cpu_records.order_record). The cooperative-write semantics mean every
    // lane writes one uint32_t; we only override lanes 0 (o_c_id) and 2
    // (o_ol_cnt) which is where these fields sit in OrderValue.
    {
        constexpr uint32_t o_c_id_offset   = offsetof(OrderValue, o_c_id)   / sizeof(uint32_t);
        constexpr uint32_t o_ol_cnt_offset = offsetof(OrderValue, o_ol_cnt) / sizeof(uint32_t);
        if (lane_id == o_c_id_offset) {
            // Deterministic logical c_id carried through indexing. (params->customer_id
            // is the GPU cache seat after remap, which is non-deterministic, so the
            // prior (customer_id % 3000)+1 derivation produced run-to-run divergence.)
            result = params->c_id;
        }
        if (lane_id == o_ol_cnt_offset) {
            result = params->num_items;
        }
    }

    gpuWriteToTableCoop(
        records.order_record, versions.order_version, params->order_id, plan->order_loc, epoch, result, lane_id,
        records.order_dirty_v1, records.order_dirty_v2);

    gpuWriteToTableCoop(records.new_order_record, versions.new_order_version, params->new_order_id, plan->new_order_loc,
        epoch, result, lane_id, records.new_order_dirty_v1, records.new_order_dirty_v2);

    for (uint32_t i = 0; i < params->num_items; ++i)
    {
        gpuReadFromTableCoop(records.item_record, versions.item_version, params->items[i].item_id,
            plan->item_plans[i].item_loc, epoch, result, lane_id);
        gpuReadFromTableCoop(records.stock_record, versions.stock_version, params->items[i].stock_id,
            plan->item_plans[i].stock_read_loc, epoch, result, lane_id);
        if (lane_id == s_quantity_offset)
        {
            uint32_t order_quantity = params->items[i].order_quantities;
            result = result > order_quantity + 10 ? result - order_quantity : result + 91 - order_quantity;
        }
        gpuWriteToTableCoop(records.stock_record, versions.stock_version, params->items[i].stock_id,
            plan->item_plans[i].stock_write_loc, epoch, result, lane_id,
            records.stock_dirty_v1, records.stock_dirty_v2);

        constexpr uint32_t ol_i_id_offset = offsetof(OrderLineValue, ol_i_id) / sizeof(uint32_t);
        constexpr uint32_t ol_amount_offset = offsetof(OrderLineValue, ol_amount) / sizeof(uint32_t);
        constexpr uint32_t ol_supply_w_id_offset = offsetof(OrderLineValue, ol_supply_w_id) / sizeof(uint32_t);
        constexpr uint32_t ol_quantity_offset = offsetof(OrderLineValue, ol_quantity) / sizeof(uint32_t);
        if (lane_id == ol_i_id_offset)
        {
            // Deterministic logical item id (items[i].item_id is the GPU cache
            // seat after remap, which is non-deterministic).
            result = params->items[i].i_id;
        }
        if (lane_id == ol_amount_offset)
        {
            result = params->items[i].order_quantities;
        }
        if (lane_id == ol_supply_w_id_offset)
        {
            // Deterministic logical supply warehouse id (params->warehouse_id is
            // the warehouse GPU seat after remap, non-deterministic at large W).
            result = params->items[i].supply_w_id;
        }
        if (lane_id == ol_quantity_offset)
        {
            result = params->items[i].order_quantities;
        }
        gpuWriteToTableCoop(records.order_line_record, versions.order_line_version, params->items[i].order_line_id,
            plan->item_plans[i].orderline_loc, epoch, result, lane_id,
            records.order_line_dirty_v1, records.order_line_dirty_v2);
    }
}

__device__ __forceinline__ void gpuExecTpccTxn(TpccRecords records, TpccVersions versions, PaymentTxnParams *params,
    PaymentTxnExecPlan *plan, uint32_t epoch, uint32_t lane_id, uint32_t txn_id)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;
    constexpr uint32_t w_ytd_offset = offsetof(WarehouseValue, w_ytd) / sizeof(uint32_t);
    constexpr uint32_t d_ytd_offset = offsetof(DistrictValue, d_ytd) / sizeof(uint32_t);
    constexpr uint32_t c_balance_offset = offsetof(CustomerValue, c_balance) / sizeof(uint32_t);
    constexpr uint32_t c_ytd_payment_offset = offsetof(CustomerValue, c_ytd_payment) / sizeof(uint32_t);
    constexpr uint32_t c_payment_cnt_offset = offsetof(CustomerValue, c_payment_cnt) / sizeof(uint32_t);


    uint32_t result;
    uint32_t payment_amount = params->payment_amount;

    gpuReadFromTableCoop(records.warehouse_record, versions.warehouse_version, params->warehouse_id,
        plan->warehouse_read_loc, epoch, result, lane_id);
    if (lane_id == w_ytd_offset)
    {
        result += payment_amount;
    }
    gpuWriteToTableCoop(records.warehouse_record, versions.warehouse_version, params->warehouse_id,
        plan->warehouse_write_loc, epoch, result, lane_id,
        records.warehouse_dirty_v1, records.warehouse_dirty_v2);

    gpuReadFromTableCoop(records.district_record, versions.district_version, params->district_id,
        plan->district_read_loc, epoch, result, lane_id);
    if (lane_id == d_ytd_offset)
    {
        result += payment_amount;
    }
    gpuWriteToTableCoop(records.district_record, versions.district_version, params->district_id,
        plan->district_write_loc, epoch, result, lane_id,
        records.district_dirty_v1, records.district_dirty_v2);

    gpuReadFromTableCoop(records.customer_record, versions.customer_version, params->customer_id,
        plan->customer_read_loc, epoch, result, lane_id);
    if (lane_id == c_balance_offset)
    {
        result -= payment_amount;
    }
    if (lane_id == c_ytd_payment_offset)
    {
        result += payment_amount;
    }
    if (lane_id == c_payment_cnt_offset)
    {
        result += 1;
    }
    gpuWriteToTableCoop(records.customer_record, versions.customer_version, params->customer_id,
        plan->customer_write_loc, epoch, result, lane_id,
        records.customer_dirty_v1, records.customer_dirty_v2);
}

__device__ __forceinline__ void gpuExecTpccTxn(TpccRecords records, TpccVersions versions, OrderStatusTxnParams *params,
    OrderStatusTxnExecPlan *plan, uint32_t epoch, uint32_t lane_id, uint32_t txn_id)
{
    uint32_t result;
    gpuReadFromTableCoop(records.customer_record, versions.customer_version, params->customer_id, plan->customer_loc,
        epoch, result, lane_id);
    gpuReadFromTableCoop(
        records.order_record, versions.order_version, params->order_id, plan->order_loc, epoch, result, lane_id);
    for (int i = 0; i < params->num_items; ++i)
    {
        gpuReadFromTableCoop(records.order_line_record, versions.order_line_version, params->orderline_ids[i],
            plan->orderline_locs[i], epoch, result, lane_id);
    }
}

void __device__ __forceinline__ gpuExecTpccTxn(TpccRecords records, TpccVersions versions, DeliveryTxnParams *params,
    DeliveryTxnExecPlan *plan, uint32_t epoch, uint32_t lane_id, uint32_t txn_id)
{
    uint32_t result;
    for (int i = 0; i < 10; ++i)
    {
        /*
        This is the cooperative memory access talked about in the paper
        This is GPU specific, as its better to load memory in big chunks, so here multiple threads read from the GPU
        These are found in gpu_storage.cuh, and the paper mentions how to read from which version.
        */
        gpuReadFromTableCoop(records.new_order_record, versions.new_order_version, params->new_order_id[i],
            plan->new_order_read_locs[i], epoch, result, lane_id);

        constexpr uint32_t o_carrier_id_offset = offsetof(OrderValue, o_carrier_id) / sizeof(uint32_t);
        gpuReadFromTableCoop(records.order_record, versions.order_version, params->order_id[i],
            plan->order_read_locs[i], epoch, result, lane_id);

        if (lane_id == o_carrier_id_offset)
        {
            result = params->carrier_id;
        }
        gpuWriteToTableCoop(records.order_record, versions.order_version, params->order_id[i],
            plan->order_write_locs[i], epoch, result, lane_id,
            records.order_dirty_v1, records.order_dirty_v2);

        constexpr uint32_t ol_amount_offset = offsetof(OrderLineValue, ol_amount) / sizeof(uint32_t);
        constexpr uint32_t ol_delivery_d_offset = offsetof(OrderLineValue, ol_delivery_d) / sizeof(uint32_t);
        uint32_t amount = 0;
        for (int j = 0; j < params->num_items[i]; ++j)
        {
            gpuReadFromTableCoop(records.order_line_record, versions.order_line_version, params->orderline_ids[i][j],
                plan->orderline_read_locs[i][j], epoch, result, lane_id);
            if (lane_id == ol_amount_offset)
            {
                amount += result;
            }
            if (lane_id == ol_delivery_d_offset)
            {
                result = params->delivery_d;
            }

            gpuWriteToTableCoop(records.order_line_record, versions.order_line_version, params->orderline_ids[i][j],
                loc_record_b, epoch, result, lane_id,
                records.order_line_dirty_v1, records.order_line_dirty_v2);

            // Delivery-aware eviction: mark this OL slot as
            // delivered so the OL stager's eviction kernel can prefer it
            // as a victim. One write per OL row (lane 0 only). The
            // pointer is null in non-hybrid modes and on stagers that did
            // not opt in, so this is a no-op there. Sentinel guard:
            // hybrid_staging's remap can leave orderline_ids at sentinel
            // when find() hit an uninitialized slot; the matching
            // gpuWriteToTableCoop already early-returns above, so this
            // separate write needs its own guard.
            if (lane_id == 0 && records.order_line_delivered_flag != nullptr
                && params->orderline_ids[i][j] != 0xffffffffu)
            {
                records.order_line_delivered_flag[params->orderline_ids[i][j]] = 1;
            }
        }

        constexpr uint32_t all_lanes_mask = 0xffffffffu;
        __shfl_sync(all_lanes_mask, amount, ol_amount_offset);

        gpuReadFromTableCoop(records.customer_record, versions.customer_version, params->customer_id[i],
            plan->customer_read_locs[i], epoch, result, lane_id);

        constexpr uint32_t c_balance_offset = offsetof(CustomerValue, c_balance) / sizeof(uint32_t);
        constexpr uint32_t c_delivery_cnt_offset = offsetof(CustomerValue, c_delivery_cnt) / sizeof(uint32_t);

        if (lane_id == c_balance_offset)
        {
            result += amount;
        }
        if (lane_id == c_delivery_cnt_offset)
        {
            ++result;
        }

        gpuWriteToTableCoop(records.customer_record, versions.customer_version, params->customer_id[i],
            plan->customer_write_locs[i], epoch, result, lane_id,
            records.customer_dirty_v1, records.customer_dirty_v2);

    }
}

void __device__ __forceinline__ gpuExecTpccTxn(TpccRecords records, TpccVersions versions, StockLevelTxnParams *params,
    StockLevelTxnExecPlan *plan, uint32_t epoch, uint32_t lane_id, uint32_t txn_id)
{
    uint32_t num_low_stock = 0;
    const uint32_t threshold = params->threshold;
    uint32_t result;
    constexpr uint32_t s_quantity_offset = offsetof(StockValue, s_quantity) / sizeof(uint32_t);
    for (uint32_t i = 0; i < params->num_items; ++i)
    {
        gpuReadFromTableCoop(records.stock_record, versions.stock_version, params->stock_ids[i],
            plan->stock_read_locs[i], epoch, result, lane_id);
        if (lane_id == s_quantity_offset && result < threshold)
        {
            ++num_low_stock;
        }
    }
    if (lane_id == s_quantity_offset)
    {
        params->num_low_stock = num_low_stock;
    }
}

union CachableTxnParams
{
    NewOrderTxnParams<FixedSizeTxn> no;
    PaymentTxnParams pmt;
    OrderStatusTxnParams os;
} __attribute__((aligned(4)));

union CachableTxnExecPlan
{
    NewOrderExecPlan<FixedSizeTxn> no;
    PaymentTxnExecPlan pmt;
    OrderStatusTxnExecPlan os;
} __attribute__((aligned(4)));

static_assert(sizeof(CachableTxnExecPlan) + sizeof(CachableTxnParams) < 1000);

template <typename GpuTxnArrayType>
__global__ void gpuExecKernel(TpccRecords records, TpccVersions versions, GpuTxnArrayType txn, GpuTxnArrayType plan,
    uint32_t num_txns, uint32_t epoch)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;

    __shared__ uint8_t cached_txn_param[num_warps][BaseTxnSize<CachableTxnParams>::value];
    __shared__ uint8_t cached_exec_plan[num_warps][BaseTxnSize<CachableTxnExecPlan>::value];
    static_assert(BaseTxnSize<CachableTxnParams>::value % sizeof(uint32_t) == 0, "Cannot be copied in 32-bit words");
    static_assert(BaseTxnSize<CachableTxnExecPlan>::value % sizeof(uint32_t) == 0, "Cannot be copied in 32-bit words");
    __shared__ uint32_t warp_counter;

    uint32_t warp_id = threadIdx.x / kDeviceWarpSize;
    uint32_t lane_id = threadIdx.x % kDeviceWarpSize;
    /* one thread loads txn id for the entire warp */
    if (threadIdx.x == 0)
    {
        warp_counter = atomicAdd(&txn_counter, num_warps);
    }

    __syncthreads();
    /* warp cooperative execution afterward */

    uint32_t warp_txn_id;
    if (lane_id == leader_lane)
    {
        warp_txn_id = atomicAdd(&warp_counter, 1);
    }
    warp_txn_id = __shfl_sync(all_lanes_mask, warp_txn_id, leader_lane);
    if (warp_txn_id >= num_txns)
    {
        return;
    }

    /* load txn param and exec plan into shared memory */
    BaseTxn *txn_param_ptr = txn.getTxn(warp_txn_id);
    BaseTxn *exec_plan_ptr = plan.getTxn(warp_txn_id);

    /* execute the txn */
    /*
    Note that we are also caching the transaction if it fits in the shared memory
    This is better for memory access because we are constantly reading from the transaction itself
    */
    switch (static_cast<TpccTxnType>((reinterpret_cast<BaseTxn *>(txn_param_ptr)->txn_type)))
    {
    case TpccTxnType::NEW_ORDER:
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_txn_param[warp_id]), reinterpret_cast<uint32_t *>(txn_param_ptr),
            BaseTxnSize<NewOrderTxnParams<FixedSizeTxn>>::value / sizeof(uint32_t), lane_id);
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_exec_plan[warp_id]), reinterpret_cast<uint32_t *>(exec_plan_ptr),
            BaseTxnSize<NewOrderExecPlan<FixedSizeTxn>>::value / sizeof(uint32_t), lane_id);
        __syncwarp();
        gpuExecTpccTxn(records, versions,
            reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(
                reinterpret_cast<BaseTxn *>(cached_txn_param[warp_id])->data),
            reinterpret_cast<NewOrderExecPlan<FixedSizeTxn> *>(
                reinterpret_cast<BaseTxn *>(cached_exec_plan[warp_id])->data),
            epoch, lane_id, warp_txn_id);
        break;
    case TpccTxnType::PAYMENT:
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_txn_param[warp_id]), reinterpret_cast<uint32_t *>(txn_param_ptr),
            BaseTxnSize<PaymentTxnParams>::value / sizeof(uint32_t), lane_id);
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_exec_plan[warp_id]), reinterpret_cast<uint32_t *>(exec_plan_ptr),
            BaseTxnSize<PaymentTxnExecPlan>::value / sizeof(uint32_t), lane_id);
        __syncwarp();
        gpuExecTpccTxn(records, versions,
            reinterpret_cast<PaymentTxnParams *>(reinterpret_cast<BaseTxn *>(cached_txn_param[warp_id])->data),
            reinterpret_cast<PaymentTxnExecPlan *>(reinterpret_cast<BaseTxn *>(cached_exec_plan[warp_id])->data), epoch,
            lane_id, warp_txn_id);
        break;
    case TpccTxnType::ORDER_STATUS:
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_txn_param[warp_id]), reinterpret_cast<uint32_t *>(txn_param_ptr),
            BaseTxnSize<OrderStatusTxnParams>::value / sizeof(uint32_t), lane_id);
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_exec_plan[warp_id]), reinterpret_cast<uint32_t *>(exec_plan_ptr),
            BaseTxnSize<OrderStatusTxnExecPlan>::value / sizeof(uint32_t), lane_id);
        __syncwarp();
        gpuExecTpccTxn(records, versions,
            reinterpret_cast<OrderStatusTxnParams *>(reinterpret_cast<BaseTxn *>(cached_txn_param[warp_id])->data),
            reinterpret_cast<OrderStatusTxnExecPlan *>(reinterpret_cast<BaseTxn *>(cached_exec_plan[warp_id])->data),
            epoch, lane_id, warp_txn_id);
        break;
    case TpccTxnType::DELIVERY:
        gpuExecTpccTxn(records, versions, reinterpret_cast<DeliveryTxnParams *>(txn_param_ptr->data),
            reinterpret_cast<DeliveryTxnExecPlan *>(exec_plan_ptr->data), epoch, lane_id, warp_txn_id);
            break;
    case TpccTxnType::STOCK_LEVEL:
        gpuExecTpccTxn(records, versions, reinterpret_cast<StockLevelTxnParams *>(txn_param_ptr->data),
            reinterpret_cast<StockLevelTxnExecPlan *>(exec_plan_ptr->data), epoch, lane_id, warp_txn_id);
        break;
    default:
        assert(false);
        break;
    }
}

} // namespace

template <typename TxnParamArrayType, typename TxnExecPlanArrayType>
void HybridExecutor<TxnParamArrayType, TxnExecPlanArrayType>::execute(uint32_t epoch)
{
    /* clear the txn_counter */
    gpu_err_check(cudaMemcpyToSymbol(txn_counter, &zero, sizeof(uint32_t)));


    uint32_t num_blocks = (config.num_txns * kDeviceWarpSize + block_size - 1) / block_size;
    gpuExecKernel<<<num_blocks, block_size>>>(
        records, versions, TpccGpuTxnArrayT(txn), TpccGpuTxnArrayT(plan), config.num_txns, epoch);
    gpu_err_check(cudaPeekAtLastError());
    gpu_err_check(cudaDeviceSynchronize());
}

template class HybridExecutor<TpccTxnParamArrayT, TpccTxnExecPlanArrayT>;

} // namespace epic::tpcc