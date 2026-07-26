//
// tpcc_hybrid_remap.cu — implementation of the per-txn-type Remap. See the
// .h for the API contract. Five per-txn-type device functions plus a top-
// level dispatch kernel that mirrors submitTpccTxn's switch-on-txn_type.
//

#include "benchmarks/tpcc_hybrid_remap.h"
#include "benchmarks/tpcc_gpu_txn.cuh"
#include "gpu_txn.cuh"
#include "util_gpu_error_check.cuh"

namespace epic::tpcc {

namespace {

// Per-table CRID→GRID lookup. Returns the grid_id stored in the flat array
// or 0xffffffff for an out-of-range CRID. Same bound semantics as the
// flat-lookup kernels in crid_grid_index.cu, inlined for use inside the
// per-txn-type kernels below where each CRID lookup is a separate field.
//
// IMPORTANT: every record_id field gets rewritten unconditionally. The
// YCSB stager's k_flat_remap_txn_record_ids skips INSERT / FULL_READ /
// FULL_READ_MODIFY_WRITE — that skip is load-bearing only for split_field's
// per-field expansion pass, which TPC-C does not have. Reintroducing the
// skip here would break the executor for any CRID that came from an INSERT
// op (NewOrder's Order / NewOrder / OrderLine writes, all of which still
// need GRID lookup so the hybrid executor can dereference the cache slot
// the stager just admitted).
__device__ __forceinline__ uint32_t lookup(const ::epic::ycsb::CridGridIndexView& v, uint32_t crid)
{
    return (crid < v.num_units) ? v.d_crid_to_grid[crid] : 0xffffffffu;
}

__device__ __forceinline__ void remapNewOrderTxn(
    NewOrderTxnParams<FixedSizeTxn>* p, const TpccCridGridViews& V)
{
    p->warehouse_id  = lookup(V.warehouse, p->warehouse_id);
    p->district_id   = lookup(V.district,  p->district_id);
    p->customer_id   = lookup(V.customer,  p->customer_id);
    p->new_order_id  = lookup(V.new_order, p->new_order_id);  // INSERT — still remapped
    p->order_id      = lookup(V.order,     p->order_id);      // INSERT — still remapped
    for (uint32_t i = 0; i < p->num_items; ++i) {
        p->items[i].item_id        = lookup(V.item,       p->items[i].item_id);
        p->items[i].stock_id       = lookup(V.stock,      p->items[i].stock_id);
        p->items[i].order_line_id  = lookup(V.order_line, p->items[i].order_line_id); // INSERT
    }
}

__device__ __forceinline__ void remapPaymentTxn(
    PaymentTxnParams* p, const TpccCridGridViews& V)
{
    p->warehouse_id = lookup(V.warehouse, p->warehouse_id);
    p->district_id  = lookup(V.district,  p->district_id);
    p->customer_id  = lookup(V.customer,  p->customer_id);
}

__device__ __forceinline__ void remapOrderStatusTxn(
    OrderStatusTxnParams* p, const TpccCridGridViews& V)
{
    p->customer_id = lookup(V.customer, p->customer_id);
    p->order_id    = lookup(V.order,    p->order_id);
    for (uint32_t i = 0; i < p->num_items; ++i) {
        p->orderline_ids[i] = lookup(V.order_line, p->orderline_ids[i]);
    }
}

__device__ __forceinline__ void remapDeliveryTxn(
    DeliveryTxnParams* p, const TpccCridGridViews& V)
{
    for (uint32_t d = 0; d < 10; ++d) {
        p->new_order_id[d] = lookup(V.new_order, p->new_order_id[d]);
        p->order_id[d]     = lookup(V.order,     p->order_id[d]);
        p->customer_id[d]  = lookup(V.customer,  p->customer_id[d]);
        for (uint32_t j = 0; j < p->num_items[d]; ++j) {
            p->orderline_ids[d][j] = lookup(V.order_line, p->orderline_ids[d][j]);
        }
    }
}

__device__ __forceinline__ void remapStockLevelTxn(
    StockLevelTxnParams* p, const TpccCridGridViews& V)
{
    for (uint32_t i = 0; i < p->num_items; ++i) {
        p->stock_ids[i] = lookup(V.stock, p->stock_ids[i]);
    }
}

template <typename GpuTxnArrayType>
__global__ void remapTpccTxnsKernel(GpuTxnArrayType txns, uint32_t num_txns,
                                    TpccCridGridViews views)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns) return;

    BaseTxn* base = txns.getTxn(tid);
    switch (static_cast<TpccTxnType>(base->txn_type)) {
    case TpccTxnType::NEW_ORDER:
        remapNewOrderTxn(reinterpret_cast<NewOrderTxnParams<FixedSizeTxn>*>(base->data), views);
        break;
    case TpccTxnType::PAYMENT:
        remapPaymentTxn(reinterpret_cast<PaymentTxnParams*>(base->data), views);
        break;
    case TpccTxnType::ORDER_STATUS:
        remapOrderStatusTxn(reinterpret_cast<OrderStatusTxnParams*>(base->data), views);
        break;
    case TpccTxnType::DELIVERY:
        remapDeliveryTxn(reinterpret_cast<DeliveryTxnParams*>(base->data), views);
        break;
    case TpccTxnType::STOCK_LEVEL:
        remapStockLevelTxn(reinterpret_cast<StockLevelTxnParams*>(base->data), views);
        break;
    default:
        // Indexer should never produce an unknown type; nothing to remap.
        break;
    }
}

} // namespace

void launchRemapTpccTxns(TpccTxnParamArrayT& params, uint32_t num_txns,
                         const TpccCridGridViews& views)
{
    if (num_txns == 0) return;
    constexpr int B = 256;
    int G = (num_txns + B - 1) / B;
    remapTpccTxnsKernel<<<G, B>>>(TpccGpuTxnArrayT(params), num_txns, views);
    gpu_err_check(cudaPeekAtLastError());
}

} // namespace epic::tpcc
