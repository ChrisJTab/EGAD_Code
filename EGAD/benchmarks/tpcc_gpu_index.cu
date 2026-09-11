//
// Created by Shujian Qian on 2023-11-20.
//

#include "tpcc_gpu_txn.cuh"

#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <memory>
#include <cuda/std/atomic>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <benchmarks/tpcc_gpu_index.h>
#include <benchmarks/tpcc_flat_index.h>
#include <benchmarks/epoch_timeline.cuh>
#include <benchmarks/tpcc_table.h>
#include <gpu_txn.cuh>
#include <util_log.h>
#include <util_gpu_error_check.cuh>

#include <cuco/static_map.cuh>

#include <cub/cub.cuh>

namespace epic::tpcc {

namespace {
__device__ __forceinline__ void tpccPrepareInsertIndex(NewOrderTxnInput<FixedSizeTxn> *txn,
    OrderKey::baseType *order_insert, NewOrderKey::baseType *new_order_insert, OrderLineKey::baseType *orderline_insert,
    uint32_t tid)
{
    uint32_t num_items = txn->num_items;
    uint32_t o_id = txn->o_id;
    uint32_t d_id = txn->d_id;
    uint32_t origin_w_id = txn->origin_w_id;
    OrderKey order_key;
    order_key.o_id = o_id;
    order_key.o_d_id = d_id;
    order_key.o_w_id = origin_w_id;
    order_insert[tid] = order_key.base_key;
    NewOrderKey new_order_key;
    new_order_key.no_o_id = o_id;
    new_order_key.no_d_id = d_id;
    new_order_key.no_w_id = origin_w_id;
    new_order_insert[tid] = new_order_key.base_key;

    OrderLineKey order_line_key;
    order_line_key.ol_o_id = o_id;
    order_line_key.ol_d_id = d_id;
    uint32_t base = tid * 15;
    for (uint32_t i = 0; i < 15; ++i)
    {
        if (i >= num_items)
        {
            orderline_insert[base + i] = static_cast<OrderLineKey::baseType>(-1);
            continue;
        }
        order_line_key.ol_w_id = txn->origin_w_id;
        order_line_key.ol_number = i + 1;
        orderline_insert[base + i] = order_line_key.base_key;
    }
}

__device__ __forceinline__ void tpccPrepareInsertIndex(PaymentTxnInput *txn, OrderKey::baseType *order_insert,
    NewOrderKey::baseType *new_order_insert, OrderLineKey::baseType *orderline_insert, uint32_t tid)
{

    order_insert[tid] = static_cast<OrderKey::baseType>(-1);
    new_order_insert[tid] = static_cast<NewOrderKey::baseType>(-1);
    for (uint32_t i = 0; i < 15; ++i)
    {
        orderline_insert[tid * 15 + i] = static_cast<OrderLineKey::baseType>(-1);
    }
}

template <typename GpuTxnArrayType>
__global__ void prepareTpccIndexKernel(GpuTxnArrayType txn, OrderKey::baseType *order_insert,
    NewOrderKey::baseType *new_order_insert, OrderLineKey::baseType *orderline_insert, uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *txn_ptr = txn.getTxn(tid);

    switch (static_cast<TpccTxnType>(txn_ptr->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        tpccPrepareInsertIndex(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn_ptr->data), order_insert,
            new_order_insert, orderline_insert, tid);
        break;
    case TpccTxnType::PAYMENT:
        tpccPrepareInsertIndex(
            reinterpret_cast<PaymentTxnInput *>(txn_ptr->data), order_insert, new_order_insert, orderline_insert, tid);
        break;
    default:
        /* No insert needed for the other three transactions */
        /* TODO: write a common funciton for these transactions */
        tpccPrepareInsertIndex(
            reinterpret_cast<PaymentTxnInput *>(txn_ptr->data), order_insert, new_order_insert, orderline_insert, tid);
        break;
    }
}

using WarehouseIndexType = cuco::static_map<WarehouseKey::baseType, uint32_t>;
using DistrictIndexType = cuco::static_map<DistrictKey::baseType, uint32_t>;
using CustomerIndexType = cuco::static_map<CustomerKey::baseType, uint32_t>;
using HistoryIndexType = cuco::static_map<HistoryKey::baseType, uint32_t>;
using NewOrderIndexType = cuco::static_map<NewOrderKey::baseType, uint32_t>;
using OrderIndexType = cuco::static_map<OrderKey::baseType, uint32_t>;
using OrderLineIndexType = cuco::static_map<OrderLineKey::baseType, uint32_t>;
using ItemIndexType = cuco::static_map<ItemKey::baseType, uint32_t>;
using StockIndexType = cuco::static_map<StockKey::baseType, uint32_t>;

using WarehouseDeviceView = WarehouseIndexType::device_view;
using DistrictDeviceView = DistrictIndexType::device_view;
using CustomerDeviceView = CustomerIndexType::device_view;
using HistoryDeviceView = HistoryIndexType::device_view;
using NewOrderDeviceView = NewOrderIndexType::device_view;
using OrderDeviceView = OrderIndexType::device_view;
using OrderLineDeviceView = OrderLineIndexType::device_view;
using ItemDeviceView = ItemIndexType::device_view;
using StockDeviceView = StockIndexType::device_view;

using epic::timeline::kAbsent;

// Per-epoch resolution state of one growing table (NewOrder, Order,
// OrderLine). A record minted this epoch is visible to a transaction only
// from the transaction that inserted it onward; a table that took the
// general path (any delete this epoch, or an insert of a key it already
// held) resolves every key against the timelines instead. Serial position
// of transaction tid: tid*4 + 0 for its inserts, + 1 for its reads and
// writes, + 2 for its deletes.
template <int PosBits>
struct GrowingTableView
{
    uint32_t minted_begin = 0;
    uint32_t num_minted = 0;
    const uint32_t* ins_entry = nullptr;   // entry of the j-th minted record
    uint32_t entries_per_txn = 1;          // entry / entries_per_txn = inserting transaction
    const uint32_t* del_pos = nullptr;     // fast delete path: earliest position deleting each record this epoch
    epic::timeline::TimelineView<PosBits> tl;   // n == 0 on the fast path

    // The record `key` resolves to before this epoch's deletes are considered.
    __device__ __forceinline__ uint32_t visible(uint32_t found, uint32_t tid) const
    {
        if (found != kAbsent && found - minted_begin < num_minted) {
            if (tid < ins_entry[found - minted_begin] / entries_per_txn) return kAbsent;
        }
        return found;
    }
    // The record `key` resolves to for transaction tid's reads and writes.
    __device__ __forceinline__ uint32_t resolve(uint64_t key, uint32_t found, uint32_t tid) const
    {
        if (tl.n != 0) return tl.resolve(key, tid * 4u + 1u, 0u, false, found);
        found = visible(found, tid);
        if (del_pos != nullptr && found != kAbsent && del_pos[found] < tid * 4u + 1u) return kAbsent;
        return found;
    }
};

template <typename View, typename KeyT>
__device__ __forceinline__ uint32_t findOrAbsent(const View& view, KeyT key)
{
    auto it = view.find(key);
    return it != view.end() ? it->second.load(cuda::std::memory_order_relaxed) : kAbsent;
}

struct tpccGpuIndexFindView
{
    WarehouseDeviceView warehouse_view;
    DistrictDeviceView district_view;
    CustomerDeviceView customer_view;
    HistoryDeviceView history_view;
    NewOrderDeviceView new_order_view;
    OrderDeviceView order_view;
    OrderLineDeviceView order_line_view;
    ItemDeviceView item_view;
    StockDeviceView stock_view;
    // Flat-OL index path: when use_flat_ol is true, the indexing
    // kernel uses order_line_flat_view instead of order_line_view for
    // every OL find. Set in TpccGpuIndex's constructor based on the
    // EPIC_FLAT_INDEX_OL env var. order_line_view is left default-
    // constructed in this case (cuco map is not allocated) and never
    // read by the kernel.
    bool use_flat_ol = false;
    OrderLineFlatView order_line_flat_view;
    // Serial-order resolution of the growing tables (see GrowingTableView).
    GrowingTableView<24> no_tv;
    GrowingTableView<24> o_tv;
    GrowingTableView<22> ol_tv;

    __device__ __forceinline__ uint32_t orderLineFind(OrderLineKey key) const
    {
        return use_flat_ol ? order_line_flat_view.find(key) : findOrAbsent(order_line_view, key.base_key);
    }
};

void __device__ __forceinline__ indexTpccTxn(NewOrderTxnInput<FixedSizeTxn> *txn,
    NewOrderTxnParams<FixedSizeTxn> *index, tpccGpuIndexFindView index_view, uint32_t tid)
{
    {
        WarehouseKey warehouse_key;
        warehouse_key.key.w_id = txn->origin_w_id;
        auto warehouse_found = index_view.warehouse_view.find(warehouse_key.base_key);
        if (warehouse_found != index_view.warehouse_view.end())
        {
            index->warehouse_id = warehouse_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        DistrictKey district_key;
        district_key.key.d_id = txn->d_id;
        district_key.key.d_w_id = txn->origin_w_id;
        auto district_found = index_view.district_view.find(district_key.base_key);
        if (district_found != index_view.district_view.end())
        {
            index->district_id = district_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
        index->next_order_id = txn->o_id;
    }

    {
        CustomerKey customer_key;
        customer_key.key.c_id = txn->c_id;
        customer_key.key.c_d_id = txn->d_id;
        customer_key.key.c_w_id = txn->origin_w_id;
        auto customer_found = index_view.customer_view.find(customer_key.base_key);
        if (customer_found != index_view.customer_view.end())
        {
            index->customer_id = customer_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    // Carry the logical c_id verbatim (data, not a key) so the executor writes a
    // deterministic o_c_id rather than the non-deterministic customer GPU seat.
    index->c_id = txn->c_id;

    {
        OrderKey order_key;
        order_key.o_id = txn->o_id;
        order_key.o_d_id = txn->d_id;
        order_key.o_w_id = txn->origin_w_id;

        index->order_id = index_view.o_tv.resolve(order_key.base_key,
            findOrAbsent(index_view.order_view, order_key.base_key), tid);
    }

    {
        NewOrderKey new_order_key;
        new_order_key.no_o_id = txn->o_id;
        new_order_key.no_d_id = txn->d_id;
        new_order_key.no_w_id = txn->origin_w_id;
        index->new_order_id = index_view.no_tv.resolve(new_order_key.base_key,
            findOrAbsent(index_view.new_order_view, new_order_key.base_key), tid);
    }

    {
        uint32_t num_items = txn->num_items;
        index->num_items = num_items;
        for (int i = 0; i < num_items; ++i)
        {
            {
                StockKey stock_key;
                stock_key.key.s_i_id = txn->items[i].i_id;
                stock_key.key.s_w_id = txn->items[i].w_id;
                auto stock_found = index_view.stock_view.find(stock_key.base_key);
                if (stock_found != index_view.stock_view.end())
                {
                    index->items[i].stock_id = stock_found->second.load(cuda::std::memory_order_relaxed);
                }
                else
                {
                    assert(false);
                }
            }
            {
                OrderLineKey order_line_key;
                order_line_key.ol_o_id = txn->o_id;
                order_line_key.ol_d_id = txn->d_id;
                order_line_key.ol_w_id = txn->origin_w_id;
                order_line_key.ol_number = i + 1;
                index->items[i].order_line_id = index_view.ol_tv.resolve(order_line_key.base_key,
                    index_view.orderLineFind(order_line_key), tid);
            }
            {
                ItemKey item_key;
                item_key.key.i_id = txn->items[i].i_id;
                auto item_found = index_view.item_view.find(item_key.base_key);
                if (item_found != index_view.item_view.end())
                {
                    index->items[i].item_id = item_found->second.load(cuda::std::memory_order_relaxed);
                }
                else
                {
                    assert(false);
                }
            }
            // Carry the non-key payload through indexing (see payment_amount
            // note above). order_quantities is data, not a key; without this
            // copy items[i].order_quantities is stale index_output buffer data,
            // corrupting the stock s_quantity update and the order_line
            // ol_amount/ol_quantity writes.
            index->items[i].order_quantities = txn->items[i].order_quantities;
            // Logical item id for the order_line ol_i_id (deterministic, vs the
            // non-deterministic item GPU seat in items[i].item_id post-remap).
            index->items[i].i_id = txn->items[i].i_id;
            // Logical supply warehouse id for the order_line ol_supply_w_id.
            index->items[i].supply_w_id = txn->items[i].w_id;
        }
    }
}

void __device__ __forceinline__ indexTpccTxn(
    PaymentTxnInput *txn, PaymentTxnParams *index, tpccGpuIndexFindView index_view, uint32_t tid)
{
    {
        WarehouseKey warehouse_key;
        warehouse_key.key.w_id = txn->warehouse_id;
        auto warehouse_found = index_view.warehouse_view.find(warehouse_key.base_key);
        if (warehouse_found != index_view.warehouse_view.end())
        {
            index->warehouse_id = warehouse_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        DistrictKey district_key;
        district_key.key.d_id = txn->district_id;
        district_key.key.d_w_id = txn->warehouse_id;
        auto district_found = index_view.district_view.find(district_key.base_key);
        if (district_found != index_view.district_view.end())
        {
            index->district_id = district_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        CustomerKey customer_key;
        customer_key.key.c_id = txn->customer_id;
        customer_key.key.c_d_id = txn->customer_district_id;
        customer_key.key.c_w_id = txn->customer_warehouse_id;
        auto customer_found = index_view.customer_view.find(customer_key.base_key);
        if (customer_found != index_view.customer_view.end())
        {
            index->customer_id = customer_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    // Carry the non-key payload through indexing. The GPU index only writes the
    // record-id fields above; payment_amount is data, not a key, so it must be
    // copied explicitly (the CPU index does this at tpcc_index.cpp). Without it,
    // PaymentTxnParams.payment_amount is left as stale index_output buffer data
    // (a prior txn's offset-12 word), so the executor adds garbage to w_ytd.
    index->payment_amount = txn->payment_amount;
}

void __device__ __forceinline__ indexTpccTxn(
    OrderStatusTxnInput *txn, OrderStatusTxnParams *index, tpccGpuIndexFindView index_view, uint32_t tid)
{
    {
        CustomerKey customer_key;
        customer_key.key.c_id = txn->c_id;
        customer_key.key.c_d_id = txn->d_id;
        customer_key.key.c_w_id = txn->w_id;
        auto customer_found = index_view.customer_view.find(customer_key.base_key);
        if (customer_found != index_view.customer_view.end())
        {
            index->customer_id = customer_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        OrderKey order_key;
        order_key.o_id = txn->o_id;
        order_key.o_d_id = txn->d_id;
        order_key.o_w_id = txn->w_id;

        index->order_id = index_view.o_tv.resolve(order_key.base_key,
            findOrAbsent(index_view.order_view, order_key.base_key), tid);
    }

    uint32_t num_items = txn->num_items;
    index->num_items = num_items;
    for (int i = 0; i < num_items; ++i)
    {
        OrderLineKey order_line_key;
        order_line_key.ol_o_id = txn->o_id;
        order_line_key.ol_d_id = txn->d_id;
        order_line_key.ol_w_id = txn->w_id;
        order_line_key.ol_number = i + 1;
        index->orderline_ids[i] = index_view.ol_tv.resolve(order_line_key.base_key,
            index_view.orderLineFind(order_line_key), tid);
    }

}

void __device__ __forceinline__ indexTpccTxn(DeliveryTxnInput *txn, DeliveryTxnParams *index, tpccGpuIndexFindView index_view, uint32_t tid)
{
    index->carrier_id = txn->carrier_id;
    index->delivery_d = txn->delivery_d;
    for (int i = 0; i < 10; ++i)
        {
            // Per-district deliver: o_id[i]==0 means this district has nothing to
            // deliver. Set the pipeline-wide invalid-CRID sentinel (the gather/
            // scatter and the executor's coop read/write already early-return on
            // it) and skip the find() so we never resolve a phantom order.
            if (txn->o_id[i] == 0)
            {
                index->new_order_id[i] = 0xffffffffu;
                continue;
            }
            NewOrderKey new_order_key;
            new_order_key.no_o_id = txn->o_id[i];
            new_order_key.no_d_id = i + 1;
            new_order_key.no_w_id = txn->w_id;
            index->new_order_id[i] = index_view.no_tv.resolve(new_order_key.base_key,
                findOrAbsent(index_view.new_order_view, new_order_key.base_key), tid);
        }

    for (int i = 0; i < 10; ++i)
        {
            if (txn->o_id[i] == 0)
            {
                index->order_id[i] = 0xffffffffu;
                continue;
            }
            OrderKey order_key;
            order_key.o_id = txn->o_id[i];
            order_key.o_d_id = i + 1;
            order_key.o_w_id = txn->w_id;
            index->order_id[i] = index_view.o_tv.resolve(order_key.base_key,
                findOrAbsent(index_view.order_view, order_key.base_key), tid);
        }

    for (int i = 0; i < 10; ++i)
        {
            // Per-district deliver skip: o_id[i]==0 means no order to deliver, so
            // txn->customers[i] was never filled by the aux index (it is the
            // invalid-CRID sentinel). Skip the find() so we do not look up a
            // phantom customer and customer_id[i] never carries a stale value.
            if (txn->o_id[i] == 0)
            {
                index->customer_id[i] = 0xffffffffu;
                continue;
            }
            CustomerKey customer_key;
            customer_key.key.c_id = txn->customers[i];
            customer_key.key.c_d_id = i + 1;
            customer_key.key.c_w_id = txn->w_id;
            auto customer_found = index_view.customer_view.find(customer_key.base_key);
            if (customer_found != index_view.customer_view.end())
            {
                index->customer_id[i] = customer_found->second.load(cuda::std::memory_order_relaxed);
            }
            else
            {
                assert(false);
            }
        }

    for (int i = 0; i < 10; ++i)
    {
        // Per-district deliver skip: no order in this district -> no order lines.
        // The aux index also forces num_items[i]=0 here, but guard explicitly so
        // a stale num_items can never drive a phantom order-line find().
        if (txn->o_id[i] == 0)
        {
            continue;
        }
        for (int j = 0; j < txn->num_items[i]; ++j)
        {
            OrderLineKey order_line_key;
            order_line_key.ol_o_id = txn->o_id[i];
            order_line_key.ol_d_id = i + 1;
            order_line_key.ol_w_id = txn->w_id;
            order_line_key.ol_number = j + 1;
            index->orderline_ids[i][j] = index_view.ol_tv.resolve(order_line_key.base_key,
                index_view.orderLineFind(order_line_key), tid);

            // if (index->orderline_ids[i][j] > 500'000)
            // {
            //     printf("txn[%d] indexed orderline[%d] > 500'000 wid[%d] did[%d] oid[%d] olid[%d]\n", tid, index->orderline_ids[i][j],
            //         txn->w_id, i + 1, txn->o_id, j + 1);
            // }
        }
    }

    index->carrier_id = txn->carrier_id;
    index->delivery_d = txn->delivery_d;

    int total_items = 0;
    for (int i = 0; i < 10; ++i)
    {
        index->num_items[i] = txn->num_items[i];
        // index->num_items[i] = 0;;
        total_items += index->num_items[i];
    }
    // printf("txn[%d] total_items[%d] ptr[%p] "
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
    //     "\n", tid, total_items, index,
    //     index->num_items[0],
    //     index->num_items[1],
    //     index->num_items[2],
    //     index->num_items[3],
    //     index->num_items[4],
    //     index->num_items[5],
    //     index->num_items[6],
    //     index->num_items[7],
    //     index->num_items[8],
    //     index->num_items[9]
    //     );

        // index->num_items[0] = txn->num_items[0];
        // index->num_items[1] = txn->num_items[1];
        // index->num_items[2] = txn->num_items[2];
        // index->num_items[3] = txn->num_items[3];
        // index->num_items[4] = txn->num_items[4];
        // index->num_items[5] = txn->num_items[5];
        // index->num_items[6] = txn->num_items[6];
        // index->num_items[7] = txn->num_items[7];
        // index->num_items[8] = txn->num_items[8];
        // index->num_items[9] = txn->num_items[9];
}

void __device__ __forceinline__ indexTpccTxn(StockLevelTxnInput *txn, StockLevelTxnParams *index, tpccGpuIndexFindView index_view, uint32_t tid)
{
    uint32_t num_unique_items = 0;
    StockKey stock_key;
    stock_key.key.s_w_id = txn->w_id;
    uint32_t prev_item = 0;
    for (int i = 0; i < txn->num_items; ++i)
    {
        uint32_t curr_item = index->stock_ids[i];  // hack, stock_ids stores item id from gpu aux index
        if (i > 0 && curr_item == prev_item)
        {
            continue;
        }
        prev_item = curr_item;
        stock_key.key.s_i_id = curr_item;
        auto stock_found = index_view.stock_view.find(stock_key.base_key);
        if (stock_found != index_view.stock_view.end())
        {
            index->stock_ids[num_unique_items] = stock_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
        ++num_unique_items;
    }
    index->num_items = num_unique_items;
    index->threshold = txn->threshold;
}

template <typename GpuTxnArrayType, typename GpuTxnIndexArrayType>
__global__ void indexTpccTxnKernel(
    GpuTxnArrayType txn, GpuTxnIndexArrayType index, tpccGpuIndexFindView index_view, uint32_t num_txns)
{

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *txn_ptr = txn.getTxn(tid);
    BaseTxn *index_ptr = index.getTxn(tid);

    index_ptr->txn_type = txn_ptr->txn_type;

    TpccTxnType txn_type = static_cast<TpccTxnType>(txn_ptr->txn_type);
    // if (txn_type == TpccTxnType::NEW_ORDER)
    // {
    //     indexTpccTxn(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn_ptr->data),
    //         reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(index_ptr->data), index_view, tid);
    // } else if (txn_type == TpccTxnType::PAYMENT)
    // {
    //     indexTpccTxn(reinterpret_cast<PaymentTxnInput *>(txn_ptr->data),
    //         reinterpret_cast<PaymentTxnParams *>(index_ptr->data), index_view, tid);
    // } else if (txn_type == TpccTxnType::ORDER_STATUS)
    // {
    //     indexTpccTxn(reinterpret_cast<OrderStatusTxnInput *>(txn_ptr->data),
    //         reinterpret_cast<OrderStatusTxnParams *>(index_ptr->data), index_view, tid);
    // } else
    //     if (txn_type == TpccTxnType::DELIVERY)
    // {
    //     indexTpccTxn(reinterpret_cast<DeliveryTxnInput *>(txn_ptr->data),
    //     reinterpret_cast<DeliveryTxnParams *>(index_ptr->data), index_view, tid);
    // }
    // else
    // {
    //         printf("tid[%d] TxnType %d not supported\n", tid, txn_ptr->txn_type);
    //         assert(false);
    // }
    //
    switch (static_cast<TpccTxnType>(txn_ptr->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        indexTpccTxn(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn_ptr->data),
            reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(index_ptr->data), index_view, tid);
        break;
    case TpccTxnType::PAYMENT:
        indexTpccTxn(reinterpret_cast<PaymentTxnInput *>(txn_ptr->data),
            reinterpret_cast<PaymentTxnParams *>(index_ptr->data), index_view, tid);
        break;
    case TpccTxnType::ORDER_STATUS:
        indexTpccTxn(reinterpret_cast<OrderStatusTxnInput *>(txn_ptr->data),
            reinterpret_cast<OrderStatusTxnParams *>(index_ptr->data), index_view, tid);
        break;
    case TpccTxnType::DELIVERY:
        indexTpccTxn(reinterpret_cast<DeliveryTxnInput *>(txn_ptr->data),
        reinterpret_cast<DeliveryTxnParams *>(index_ptr->data), index_view, tid);
        break;
    case TpccTxnType::STOCK_LEVEL:
        indexTpccTxn(reinterpret_cast<StockLevelTxnInput *>(txn_ptr->data),
        reinterpret_cast<StockLevelTxnParams *>(index_ptr->data), index_view, tid);
        break;
    default:
            assert(false);
        break;
    }
}

// Timeline inputs of the three growing tables, one entry layout per table:
// NewOrder 11 entries per transaction (tid*11 = the NewOrder insert at
// position tid*4; tid*11+1+i = the Delivery delete of district i at
// position tid*4+2), Order 1 entry per transaction (tid, position tid*4),
// OrderLine 15 entries per transaction (tid*15+i, position tid*4). Every
// pos entry is written; the key arrays were cleared to the sentinel by the
// caller, so only real inserts and deletes are set.
template <typename GpuTxnArrayType>
__global__ void k_fill_growing_timeline_inputs(GpuTxnArrayType txn, uint32_t num_txns,
    NewOrderKey::baseType* no_ins, NewOrderKey::baseType* no_del, uint32_t* no_pos,
    OrderKey::baseType* o_ins, uint32_t* o_pos,
    OrderLineKey::baseType* ol_ins, uint32_t* ol_pos)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns) return;
    BaseTxn *txn_ptr = txn.getTxn(tid);
    constexpr auto kNoNo = static_cast<NewOrderKey::baseType>(-1);
    constexpr auto kNoO  = static_cast<OrderKey::baseType>(-1);
    constexpr auto kNoOl = static_cast<OrderLineKey::baseType>(-1);
    if (no_pos) {
        no_pos[tid * 11] = tid * 4;
        for (int i = 0; i < 10; ++i) no_pos[tid * 11 + 1 + i] = tid * 4 + 2;
        for (int i = 0; i < 11; ++i) { no_ins[tid * 11 + i] = kNoNo; no_del[tid * 11 + i] = kNoNo; }
    }
    if (o_pos) { o_pos[tid] = tid * 4; o_ins[tid] = kNoO; }
    if (ol_pos) for (int i = 0; i < 15; ++i) { ol_pos[tid * 15 + i] = tid * 4; ol_ins[tid * 15 + i] = kNoOl; }
    switch (static_cast<TpccTxnType>(txn_ptr->txn_type))
    {
    case TpccTxnType::NEW_ORDER: {
        auto *in = reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn_ptr->data);
        if (no_ins) {
            NewOrderKey k; k.no_o_id = in->o_id; k.no_d_id = in->d_id; k.no_w_id = in->origin_w_id;
            no_ins[tid * 11] = k.base_key;
        }
        if (o_ins) {
            OrderKey k; k.o_id = in->o_id; k.o_d_id = in->d_id; k.o_w_id = in->origin_w_id;
            o_ins[tid] = k.base_key;
        }
        if (ol_ins) {
            for (uint32_t i = 0; i < in->num_items && i < 15; ++i) {
                OrderLineKey k; k.ol_o_id = in->o_id; k.ol_d_id = in->d_id; k.ol_w_id = in->origin_w_id;
                k.ol_number = i + 1;
                ol_ins[tid * 15 + i] = k.base_key;
            }
        }
        break;
    }
    case TpccTxnType::DELIVERY: {
        auto *in = reinterpret_cast<DeliveryTxnInput *>(txn_ptr->data);
        if (no_del) {
            for (int i = 0; i < 10; ++i) {
                if (in->o_id[i] == 0) continue;
                NewOrderKey k; k.no_o_id = in->o_id[i]; k.no_d_id = i + 1; k.no_w_id = in->w_id;
                no_del[tid * 11 + 1 + i] = k.base_key;
            }
        }
        break;
    }
    default:
        break;
    }
}

// NewOrder fast delete path, pass 1: the earliest position (tid*4+2, the
// deleting transaction's delete class) at which each record is delivered
// this epoch; a delivery of a record that is not visible to the
// transaction (never inserted, or inserted later) is a no-op.
template <typename GpuTxnArrayType>
__global__ void k_record_no_deletes(GpuTxnArrayType txn, tpccGpuIndexFindView view, uint32_t num_txns,
                                    uint32_t* __restrict__ del_pos)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns) return;
    BaseTxn *txn_ptr = txn.getTxn(tid);
    if (static_cast<TpccTxnType>(txn_ptr->txn_type) != TpccTxnType::DELIVERY) return;
    auto *in = reinterpret_cast<DeliveryTxnInput *>(txn_ptr->data);
    for (int i = 0; i < 10; ++i) {
        if (in->o_id[i] == 0) continue;
        NewOrderKey k; k.no_o_id = in->o_id[i]; k.no_d_id = i + 1; k.no_w_id = in->w_id;
        const uint32_t c = view.no_tv.visible(findOrAbsent(view.new_order_view, k.base_key), tid);
        if (c != kAbsent) atomicMin(&del_pos[c], tid * 4u + 2u);
    }
}

// NewOrder fast delete path, pass 3 (after the lookups): the effective
// deletes, one flag per slot (tid*10 + district) with the key and the
// ended CRID; the delete log's order.
template <typename GpuTxnArrayType, typename GpuTxnIndexArrayType>
__global__ void k_no_effective_deletes(GpuTxnArrayType txn, GpuTxnIndexArrayType index, uint32_t num_txns,
                                       const uint32_t* __restrict__ del_pos,
                                       NewOrderKey::baseType* __restrict__ keys, uint32_t* __restrict__ crids,
                                       uint8_t* __restrict__ flags)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns) return;
    BaseTxn *txn_ptr = txn.getTxn(tid);
    const int base = tid * 10;
    if (static_cast<TpccTxnType>(txn_ptr->txn_type) != TpccTxnType::DELIVERY) {
        for (int i = 0; i < 10; ++i) flags[base + i] = 0;
        return;
    }
    auto *in = reinterpret_cast<DeliveryTxnInput *>(txn_ptr->data);
    auto *params = reinterpret_cast<DeliveryTxnParams *>(index.getTxn(tid)->data);
    for (int i = 0; i < 10; ++i) {
        const uint32_t c = params->new_order_id[i];
        const bool eff = in->o_id[i] != 0 && c != kAbsent && del_pos[c] == tid * 4u + 2u;
        flags[base + i] = eff ? 1 : 0;
        if (eff) {
            NewOrderKey k; k.no_o_id = in->o_id[i]; k.no_d_id = i + 1; k.no_w_id = in->w_id;
            keys[base + i] = k.base_key;
            crids[base + i] = c;
        }
    }
}

__global__ void k_reset_del_pos(const uint32_t* __restrict__ crids, uint32_t n, uint32_t* __restrict__ del_pos)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    del_pos[crids[i]] = kAbsent;
}

// Flat OL index helpers for the timeline path: epoch-start lookup of the
// event keys, and the final-life assignment at the epoch boundary.
__global__ void k_flat_ol_find_many(const OrderLineKey::baseType* __restrict__ keys, uint32_t n,
                                    OrderLineFlatView view, uint32_t* __restrict__ out)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    OrderLineKey k; k.base_key = keys[i];
    out[i] = view.find(k);
}
__global__ void k_flat_ol_assign_many(const OrderLineKey::baseType* __restrict__ keys,
                                      const uint32_t* __restrict__ crids, uint32_t n, OrderLineFlatView view)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    OrderLineKey k; k.base_key = keys[i];
    view.insert(k, crids[i]);
}
// Flat OL bulk insert that reports keys already present (a second insert
// of a key the table holds), so the epoch can fall back to the timeline.
__global__ void k_flat_ol_bulk_insert_checked(const OrderLineKey::baseType* __restrict__ keys,
    const uint32_t* __restrict__ values, uint32_t n, OrderLineFlatView view, uint32_t* __restrict__ rejected,
    uint32_t* __restrict__ out_of_range)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    OrderLineKey k; k.base_key = keys[i];
    if (k.ol_o_id > view.max_o) { atomicAdd(out_of_range, 1u); return; }
    uint32_t* slot = view.d_array + denseIdxOL(k, view.max_o);
    if (atomicCAS(slot, kAbsent, values[i]) != kAbsent) atomicAdd(rejected, 1u);
}

template<typename InputType>
class DummyPredicate
{
public:
    __device__ __forceinline__ bool operator()(InputType val)
    {
        return true;
    }
};

} // namespace

template <typename TxnArrayType, typename TxnParamArrayType, typename GpuTxnArrayType>
class TpccGpuIndexImpl
{
public:
    static constexpr double load_factor = 0.5;

    static constexpr cuco::empty_key<WarehouseKey::baseType> warehouse_key_sentinel{
        static_cast<WarehouseKey::baseType>(-1)};
    static constexpr cuco::empty_key<DistrictKey::baseType> district_key_sentinel{
        static_cast<DistrictKey::baseType>(-1)};
    static constexpr cuco::empty_key<CustomerKey::baseType> customer_key_sentinel{
        static_cast<CustomerKey::baseType>(-1)};
    static constexpr cuco::empty_key<HistoryKey::baseType> history_key_sentinel{static_cast<HistoryKey::baseType>(-1)};
    static constexpr cuco::empty_key<NewOrderKey::baseType> new_order_key_sentinel{
        static_cast<NewOrderKey::baseType>(-1)};
    static constexpr cuco::empty_key<OrderKey::baseType> order_key_sentinel{static_cast<OrderKey::baseType>(-1)};
    static constexpr cuco::empty_key<OrderLineKey::baseType> order_line_key_sentinel{
        static_cast<OrderLineKey::baseType>(-1)};
    static constexpr cuco::empty_key<ItemKey::baseType> item_key_sentinel{static_cast<ItemKey::baseType>(-1)};
    static constexpr cuco::empty_key<StockKey::baseType> stock_key_sentinel{static_cast<StockKey::baseType>(-1)};

    static constexpr cuco::empty_value<uint32_t> value_sentinel{static_cast<uint32_t>(-1)};

    // Erased-slot sentinel for the NewOrder map, the one TPC-C table with
    // deletes (Delivery consumes each NewOrder row exactly once). Passed
    // at BOTH construction sites (ctor and the recovery rebuild); cuco's
    // erase throws at runtime on a map built without it. Outside the key
    // universe: real packed keys leave the upper bits zero.
    static constexpr cuco::erased_key<NewOrderKey::baseType> new_order_erased_sentinel{
        static_cast<NewOrderKey::baseType>(-2)};

    TpccConfig tpcc_config;

    // === CPU shadow index reference ===
    //
    // The host-side shadow (8 vectors, max_o stride, rollback snapshots,
    // per-epoch insert-key D2H buffers) lives in TpccDb as a
    // TpccCpuShadowIndex. This class holds a reference. The shadow is
    // the host-side ground truth: after a crash, recovery repopulates a
    // fresh GPU index from it. The validator, the per-epoch mirror call,
    // the rollback accessor, and the rebuild upload all read through
    // shadow_. Declared here (next to tpcc_config, before the cuco maps)
    // so the ctor's member-init list order matches declaration order.
    TpccCpuShadowIndex& shadow_;

    std::shared_ptr<WarehouseIndexType> warehouse_index;
    std::shared_ptr<DistrictIndexType> district_index;
    std::shared_ptr<CustomerIndexType> customer_index;
    std::shared_ptr<HistoryIndexType> history_index;
    std::shared_ptr<NewOrderIndexType> new_order_index;
    std::shared_ptr<OrderIndexType> order_index;
    std::shared_ptr<OrderLineIndexType> order_line_index;
    std::shared_ptr<ItemIndexType> item_index;
    std::shared_ptr<StockIndexType> stock_index;
    tpccGpuIndexFindView index_device_view; /* the order is crucial so that it's initialized after the indices */

    uint32_t *d_order_free_rows;
    uint32_t *d_new_order_free_rows;
    uint32_t *d_order_line_free_rows;
    thrust::device_ptr<uint32_t> dp_order_free_rows;
    thrust::device_ptr<uint32_t> dp_new_order_free_rows;
    thrust::device_ptr<uint32_t> dp_order_line_free_rows;
    uint32_t order_free_start = 0;
    uint32_t new_order_free_start = 0;
    uint32_t order_line_free_start = 0;

    OrderKey::baseType *d_order_insert, *d_order_valid_insert;
    NewOrderKey::baseType *d_new_order_insert, *d_new_order_valid_insert;
    OrderLineKey::baseType *d_order_line_insert, *d_order_line_valid_insert;
    thrust::device_ptr<OrderKey::baseType> dp_order_valid_insert;
    thrust::device_ptr<NewOrderKey::baseType> dp_new_order_valid_insert;
    thrust::device_ptr<OrderLineKey::baseType> dp_order_line_valid_insert;
    uint32_t *d_order_num_insert, *d_new_order_num_insert, *d_order_line_num_insert;

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Serial-order resolution of the growing tables (epoch_timeline.cuh).
    // All three take the fast path (bulk insert; NewOrder additionally
    // records the first delete position of every delivered record) and
    // fall back to the timeline when an insert is rejected because the key
    // is already held. Each table also records the entry of every record it
    // minted this epoch (fast path), the visibility bound of that record.
    epic::timeline::EpochTimeline<NewOrderKey::baseType, 24> no_tl_;
    epic::timeline::EpochTimeline<OrderKey::baseType, 24> o_tl_;
    epic::timeline::EpochTimeline<OrderLineKey::baseType, 22> ol_tl_;
    uint32_t *d_no_ins_entry = nullptr, *d_o_ins_entry = nullptr, *d_ol_ins_entry = nullptr;
    uint32_t *d_flat_rejected = nullptr, *h_flat_rejected = nullptr;   // mapped: [0] rejected (key held), [1] out of range
    uint32_t no_delete_count = 0;          // cumulative; the durable NO delete-log cursor
    uint32_t num_no_deletes_this_epoch = 0;
    bool no_general_ = false, o_general_ = false, ol_general_ = false;   // this epoch's paths
    // NewOrder fast delete path (Delivery-bearing hybrid runs without a
    // rejected insert): del_pos per NO record (allocated on first use), the
    // per-slot effective-delete columns (tid*10 + district) and their
    // compactions in slot order (the delete log's order).
    uint32_t *d_no_del_pos = nullptr;
    NewOrderKey::baseType *d_no_edel_key = nullptr, *d_no_valid_deletes = nullptr;
    uint32_t *d_no_edel_crid = nullptr, *d_no_valid_delete_crids = nullptr;
    uint8_t *d_no_edel_flag = nullptr;
    thrust::device_ptr<NewOrderKey::baseType> dp_no_valid_deletes;
    uint32_t *d_num_no_delete = nullptr, *h_num_no_delete = nullptr;   // mapped
    void *d_no_flagged_temp = nullptr;
    size_t no_flagged_temp_bytes = 0;
    const uint32_t *d_no_edel_out = nullptr;   // this epoch's effective-delete CRIDs (fast or timeline path)

    // Flat OL index. When EPIC_FLAT_INDEX_OL=1, allocate
    // a uint32_t flat array sized to W * 10 * max_o * 15 entries instead
    // of (or alongside) the cuco hash map. Cuco is allocated at a tiny
    // sentinel size (1024 entries -> 2048 slots at load_factor 0.5, ~32 KB)
    // when flat is on, so the init list stays simple; the find/insert paths
    // route to the flat array.
    bool use_flat_ol_index_ = false;
    uint32_t *d_order_line_flat_ = nullptr;
    uint32_t  order_line_flat_max_o_ = 0;
    uint64_t  order_line_flat_size_  = 0;  // total slots in the flat array

    // OL cuco capacity. When the flat-OL path is active, the cuco map
    // is unused but still constructed (init list cannot skip it), so
    // size it down to ~32 KB. The flatOLEnabled() / computeOLMaxO()
    // helpers now live in tpcc_flat_index.h so the shadow index and the
    // GPU index agree on the OL dense stride.
    static size_t cucoOLCapacity(const TpccConfig& cfg) {
        return flatOLEnabled(cfg.execution_mode)
            ? static_cast<size_t>(std::ceil(1024 / load_factor))
            : static_cast<size_t>(std::ceil(cfg.orderLineTableSize() / load_factor));
    }

    explicit TpccGpuIndexImpl(TpccConfig tpcc_config, TpccCpuShadowIndex& shadow)
        : tpcc_config(tpcc_config)
        , shadow_(shadow)
        , warehouse_index{std::make_shared<WarehouseIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.warehouseTableSize() / load_factor)), warehouse_key_sentinel,
              value_sentinel)}
        , district_index{std::make_shared<DistrictIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.districtTableSize() / load_factor)), district_key_sentinel,
              value_sentinel)}
        , customer_index{std::make_shared<CustomerIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.customerTableSize() / load_factor)), customer_key_sentinel,
              value_sentinel)}
        , history_index{std::make_shared<HistoryIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.historyTableSize() / load_factor)), history_key_sentinel,
              value_sentinel)}
        , new_order_index{std::make_shared<NewOrderIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.newOrderTableSize() / load_factor)), new_order_key_sentinel,
              value_sentinel, new_order_erased_sentinel)}
        , order_index{std::make_shared<OrderIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.orderTableSize() / load_factor)), order_key_sentinel,
              value_sentinel)}
        , order_line_index{std::make_shared<OrderLineIndexType>(
              cucoOLCapacity(tpcc_config), order_line_key_sentinel,
              value_sentinel)}
        , item_index{std::make_shared<ItemIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.itemTableSize() / load_factor)), item_key_sentinel,
              value_sentinel)}
        , stock_index{std::make_shared<StockIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.stockTableSize() / load_factor)), stock_key_sentinel,
              value_sentinel)}
        , index_device_view{warehouse_index->get_device_view(), district_index->get_device_view(),
              customer_index->get_device_view(), history_index->get_device_view(), new_order_index->get_device_view(),
              order_index->get_device_view(), order_line_index->get_device_view(), item_index->get_device_view(),
              stock_index->get_device_view()}
    {
        auto &logger = Logger::GetInstance();

        // Flat OL index: if env-gated on, allocate the
        // flat array and patch the device view to use it. Cuco was
        // already constructed at sentinel size (32 KB) above; it stays
        // unused in this mode.
        use_flat_ol_index_ = flatOLEnabled(tpcc_config.execution_mode);
        if (use_flat_ol_index_) {
            order_line_flat_max_o_ = computeOLMaxO(tpcc_config);
            order_line_flat_size_ = static_cast<uint64_t>(tpcc_config.num_warehouses)
                                  * 10ull
                                  * static_cast<uint64_t>(order_line_flat_max_o_)
                                  * 15ull;
            const size_t bytes = order_line_flat_size_ * sizeof(uint32_t);
            gpu_err_check(cudaMalloc(&d_order_line_flat_, bytes));
            // Initialize to sentinel so a stray find before insert returns -1
            // rather than zero (which is a valid CRID).
            gpu_err_check(cudaMemset(d_order_line_flat_, 0xff, bytes));
            index_device_view.use_flat_ol = true;
            index_device_view.order_line_flat_view =
                OrderLineFlatView{d_order_line_flat_, order_line_flat_max_o_};
            logger.Info("[FLAT-OL] OL flat index active: "
                        "max_o={} size={} entries ({} MB)",
                        order_line_flat_max_o_, order_line_flat_size_,
                        bytes / (1024 * 1024));
        }

        gpu_err_check(cudaMalloc(&d_order_free_rows, tpcc_config.orderTableSize() * sizeof(uint32_t)));
        dp_order_free_rows = thrust::device_pointer_cast(d_order_free_rows);
        gpu_err_check(cudaMalloc(&d_new_order_free_rows, tpcc_config.newOrderTableSize() * sizeof(uint32_t)));
        dp_new_order_free_rows = thrust::device_pointer_cast(d_new_order_free_rows);
        gpu_err_check(cudaMalloc(&d_order_line_free_rows, tpcc_config.orderLineTableSize() * sizeof(uint32_t)));
        dp_order_line_free_rows = thrust::device_pointer_cast(d_order_line_free_rows);

        gpu_err_check(cudaMalloc(&d_order_insert, tpcc_config.num_txns * sizeof(OrderKey::baseType)));
        gpu_err_check(cudaMalloc(&d_order_valid_insert, tpcc_config.num_txns * sizeof(OrderKey::baseType)));
        dp_order_valid_insert = thrust::device_pointer_cast(d_order_valid_insert);
        gpu_err_check(cudaMalloc(&d_new_order_insert, tpcc_config.num_txns * sizeof(NewOrderKey::baseType)));
        gpu_err_check(cudaMalloc(&d_new_order_valid_insert, tpcc_config.num_txns * sizeof(NewOrderKey::baseType)));
        dp_new_order_valid_insert = thrust::device_pointer_cast(d_new_order_valid_insert);
        gpu_err_check(cudaMalloc(&d_order_line_insert, tpcc_config.num_txns * 15 * sizeof(OrderLineKey::baseType)));
        gpu_err_check(
            cudaMalloc(&d_order_line_valid_insert, tpcc_config.num_txns * 15 * sizeof(OrderLineKey::baseType)));
        dp_order_line_valid_insert = thrust::device_pointer_cast(d_order_line_valid_insert);
        gpu_err_check(cudaMalloc(&d_order_num_insert, sizeof(uint32_t)));
        gpu_err_check(cudaMalloc(&d_new_order_num_insert, sizeof(uint32_t)));
        gpu_err_check(cudaMalloc(&d_order_line_num_insert, sizeof(uint32_t)));

        size_t max_bytes = 0;
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_order_insert, d_order_valid_insert,
            d_order_num_insert, tpcc_config.num_txns, DummyPredicate<OrderKey::baseType>());
        max_bytes = std::max(max_bytes, temp_storage_bytes);

        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_new_order_insert, d_new_order_valid_insert,
            d_new_order_num_insert, tpcc_config.num_txns, DummyPredicate<NewOrderKey::baseType>());
        max_bytes = std::max(max_bytes, temp_storage_bytes);

        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_order_line_insert, d_order_line_valid_insert,
            d_order_line_num_insert, tpcc_config.num_txns * 15, DummyPredicate<OrderLineKey::baseType>());
        max_bytes = std::max(max_bytes, temp_storage_bytes);

        temp_storage_bytes = max_bytes;
        logger.Trace("Allocating {} bytes for temp storage", formatSizeBytes(temp_storage_bytes));
        gpu_err_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // Minted-record entries (fast path) and the NewOrder timeline for
        // Delivery-bearing hybrid_staging runs. The delete path is part of
        // EGAD's staging design; the cpu_only / gpu_only baseline modes stay
        // pure Epic (which performs no deletes), keeping their throughput
        // floors and the upstream-parity claim untouched. The Order and
        // OrderLine timelines are allocated on first use.
        gpu_err_check(cudaMalloc(&d_no_ins_entry, tpcc_config.num_txns * sizeof(uint32_t)));
        gpu_err_check(cudaMalloc(&d_o_ins_entry, tpcc_config.num_txns * sizeof(uint32_t)));
        gpu_err_check(cudaMalloc(&d_ol_ins_entry, tpcc_config.num_txns * 15 * sizeof(uint32_t)));
        gpu_err_check(cudaHostAlloc(&h_flat_rejected, 2 * sizeof(uint32_t), cudaHostAllocMapped));
        h_flat_rejected[0] = 0; h_flat_rejected[1] = 0;
        gpu_err_check(cudaHostGetDevicePointer(&d_flat_rejected, h_flat_rejected, 0));
        {
            size_t b = 0;
            cub::DeviceSelect::If(nullptr, b, thrust::counting_iterator<uint32_t>(0), d_ol_ins_entry,
                d_order_line_num_insert, tpcc_config.num_txns * 15, epic::timeline::detail::EntryHasKey<OrderLineKey::baseType>{d_order_line_insert});
            if (b > temp_storage_bytes) {
                gpu_err_check(cudaFree(d_temp_storage));
                temp_storage_bytes = b;
                gpu_err_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            }
        }
        if (tpcc_config.execution_mode == ExecMode::HYBRID_STAGING && tpcc_config.txn_mix.delivery > 0) {
            const size_t n_slots = static_cast<size_t>(tpcc_config.num_txns) * 10;
            gpu_err_check(cudaMalloc(&d_no_edel_key, sizeof(NewOrderKey::baseType) * n_slots));
            gpu_err_check(cudaMalloc(&d_no_valid_deletes, sizeof(NewOrderKey::baseType) * n_slots));
            gpu_err_check(cudaMalloc(&d_no_edel_crid, sizeof(uint32_t) * n_slots));
            gpu_err_check(cudaMalloc(&d_no_valid_delete_crids, sizeof(uint32_t) * n_slots));
            gpu_err_check(cudaMalloc(&d_no_edel_flag, sizeof(uint8_t) * n_slots));
            dp_no_valid_deletes = thrust::device_pointer_cast(d_no_valid_deletes);
            gpu_err_check(cudaHostAlloc(&h_num_no_delete, sizeof(uint32_t), cudaHostAllocMapped));
            *h_num_no_delete = 0;
            gpu_err_check(cudaHostGetDevicePointer(&d_num_no_delete, h_num_no_delete, 0));
            size_t key_bytes = 0, crid_bytes = 0;
            cub::DeviceSelect::Flagged(nullptr, key_bytes,
                d_no_edel_key, d_no_edel_flag, d_no_valid_deletes, d_num_no_delete, n_slots);
            cub::DeviceSelect::Flagged(nullptr, crid_bytes,
                d_no_edel_crid, d_no_edel_flag, d_no_valid_delete_crids, d_num_no_delete, n_slots);
            no_flagged_temp_bytes = std::max(key_bytes, crid_bytes);
            gpu_err_check(cudaMalloc(&d_no_flagged_temp, no_flagged_temp_bytes));
        }

        // CPU shadow allocation now lives in TpccCpuShadowIndex's ctor
        // (host-side, owned by TpccDb). This class just holds a reference.

        logger.Info("Finished constructing TpccGpuIndex");
        size_t free, total;
        gpu_err_check(cudaMemGetInfo(&free, &total));
        logger.Info("GPU memory usage: {} / {}", formatSizeBytes(total - free), formatSizeBytes(total));
    }
    void loadInitialData()
    {
        auto &logger = Logger::GetInstance();
        logger.Info("Loading initial data");

        {

            logger.Trace("Loading warehouse table");
            std::vector<WarehouseKey::baseType> warehouse_keys;
            warehouse_keys.reserve(tpcc_config.num_warehouses);
            for (uint32_t i = 0; i < tpcc_config.num_warehouses; ++i)
            {
                warehouse_keys.push_back(WarehouseKey{i + 1}.base_key);
            }
            thrust::device_vector<WarehouseKey::baseType> d_warehouse_keys(warehouse_keys);
            thrust::device_vector<uint32_t> d_warehouse_values(tpcc_config.num_warehouses);
            thrust::sequence(d_warehouse_values.begin(), d_warehouse_values.end(), 0);
            auto zipped_warehouse_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_warehouse_keys.begin(), d_warehouse_values.begin()));
            warehouse_index->insert(zipped_warehouse_kv, zipped_warehouse_kv + tpcc_config.num_warehouses);
        }

        {
            logger.Trace("Loading district table");
            size_t num_districts = tpcc_config.num_warehouses * 10;
            std::vector<DistrictKey::baseType> district_keys;
            district_keys.reserve(num_districts);
            for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
            {
                for (uint32_t d_id = 1; d_id <= 10; ++d_id)
                {
                    district_keys.push_back(DistrictKey{d_id, w_id}.base_key);
                }
            }
            thrust::device_vector<DistrictKey::baseType> d_district_keys(district_keys);
            thrust::device_vector<uint32_t> d_district_values(num_districts);
            thrust::sequence(d_district_values.begin(), d_district_values.end(), 0);
            auto zipped_district_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_district_keys.begin(), d_district_values.begin()));
            district_index->insert(zipped_district_kv, zipped_district_kv + num_districts);
        }

        {
            logger.Trace("Loading customer table");
            size_t num_customers = tpcc_config.num_warehouses * 10 * 3000;
            std::vector<CustomerKey::baseType> customer_keys;
            customer_keys.reserve(num_customers);
            for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
            {
                for (uint32_t d_id = 1; d_id <= 10; ++d_id)
                {
                    for (uint32_t c_id = 1; c_id <= 3000; ++c_id)
                    {
                        customer_keys.push_back(CustomerKey{c_id, d_id, w_id}.base_key);
                    }
                }
            }
            thrust::device_vector<CustomerKey::baseType> d_customer_keys(customer_keys);
            thrust::device_vector<uint32_t> d_customer_values(num_customers);
            thrust::sequence(d_customer_values.begin(), d_customer_values.end(), 0);
            auto zipped_customer_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_customer_keys.begin(), d_customer_values.begin()));
            customer_index->insert(zipped_customer_kv, zipped_customer_kv + num_customers);
        }

        {
            /* TODO: populate data in History Table */
        }

        {
            logger.Trace("Loading item table");
            size_t num_items = 100'000;
            std::vector<ItemKey::baseType> item_keys;
            item_keys.reserve(num_items);
            for (uint32_t i_id = 1; i_id <= 100'000; ++i_id)
            {
                item_keys.push_back(ItemKey{i_id}.base_key);
            }
            thrust::device_vector<ItemKey::baseType> d_item_keys(item_keys);
            thrust::device_vector<uint32_t> d_item_values(num_items);
            thrust::sequence(d_item_values.begin(), d_item_values.end(), 0);
            auto zipped_item_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_item_keys.begin(), d_item_values.begin()));
            item_index->insert(zipped_item_kv, zipped_item_kv + num_items);
        }

        {
            logger.Trace("Loading stock table");
            size_t num_stocks = tpcc_config.num_warehouses * 100'000;
            std::vector<StockKey::baseType> stock_keys;
            stock_keys.reserve(num_stocks);
            for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
            {
                for (uint32_t i_id = 1; i_id <= 100'000; ++i_id)
                {
                    stock_keys.push_back(StockKey{i_id, w_id}.base_key);
                }
            }
            thrust::device_vector<StockKey::baseType> d_stock_keys(stock_keys);
            thrust::device_vector<uint32_t> d_stock_values(num_stocks);
            thrust::sequence(d_stock_values.begin(), d_stock_values.end(), 0);
            auto zipped_stock_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_stock_keys.begin(), d_stock_values.begin()));
            stock_index->insert(zipped_stock_kv, zipped_stock_kv + num_stocks);
        }

        {
            logger.Trace("Loading order table");
            size_t num_orders = tpcc_config.num_warehouses * 10 * 3'000;
            size_t num_new_orders = tpcc_config.num_warehouses * 10 * 900;
            size_t num_order_lines = tpcc_config.num_warehouses * 10 * 3'000 * 15;
            std::vector<OrderKey::baseType> order_keys;
            std::vector<NewOrderKey::baseType> new_order_keys;
            std::vector<OrderLineKey::baseType> order_line_keys;
            order_keys.reserve(num_orders);
            new_order_keys.reserve(num_new_orders);
            // Skip building the OL key vector when the flat-index path
            // is active; the init kernel computes (w, d, o, ol) from
            // its thread index instead. Avoids a 1.5 GB host+device
            // transient at W=128 E=200.
            if (!use_flat_ol_index_) {
                order_line_keys.reserve(num_order_lines);
            }
            for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
            {
                for (uint32_t d_id = 1; d_id <= 10; ++d_id)
                {
                    for (uint32_t o_id = 1; o_id <= 3'000; ++o_id)
                    {
                        order_keys.push_back(OrderKey{o_id, d_id, w_id}.base_key);
                        if (o_id > 2'100)
                        {
                            new_order_keys.push_back(NewOrderKey{o_id, d_id, w_id}.base_key);
                        }
                        if (!use_flat_ol_index_) {
                            for (uint32_t ol_number = 1; ol_number <= 15; ++ol_number)
                            {
                                order_line_keys.push_back(OrderLineKey{o_id, d_id, w_id, ol_number}.base_key);
                            }
                        }
                    }
                }
            }

            thrust::device_vector<OrderKey::baseType> d_order_keys(order_keys);
            thrust::device_vector<NewOrderKey::baseType> d_new_order_keys(new_order_keys);

            thrust::device_vector<uint32_t> d_order_values(num_orders);
            thrust::device_vector<uint32_t> d_new_order_values(num_new_orders);

            thrust::sequence(d_order_values.begin(), d_order_values.end(), 0);
            thrust::sequence(d_new_order_values.begin(), d_new_order_values.end(), 0);

            auto zipped_order_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_order_keys.begin(), d_order_values.begin()));
            auto zipped_new_order_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_new_order_keys.begin(), d_new_order_values.begin()));

            order_index->insert(zipped_order_kv, zipped_order_kv + num_orders);
            new_order_index->insert(zipped_new_order_kv, zipped_new_order_kv + num_new_orders);
            if (use_flat_ol_index_) {
                // Transient-free path: the init kernel encodes the
                // (w, d, o, ol) -> dense_idx mapping internally and
                // writes flat[dense_idx] = tid. No keys/values arrays
                // needed.
                constexpr int B = 256;
                uint32_t total_initial_ol =
                    static_cast<uint32_t>(tpcc_config.num_warehouses) * 10u * 3000u * 15u;
                int G = (total_initial_ol + B - 1) / B;
                k_flat_ol_init_initial_pop<<<G, B>>>(
                    d_order_line_flat_,
                    static_cast<uint32_t>(tpcc_config.num_warehouses),
                    order_line_flat_max_o_);
                gpu_err_check(cudaPeekAtLastError());
                gpu_err_check(cudaDeviceSynchronize());
            } else {
                // Legacy cuco path: build the OL key+value device vectors
                // here (kept inside the else so they aren't allocated when
                // flat is on).
                thrust::device_vector<OrderLineKey::baseType> d_order_line_keys(order_line_keys);
                thrust::device_vector<uint32_t> d_order_line_values(num_order_lines);
                thrust::sequence(d_order_line_values.begin(), d_order_line_values.end(), 0);
                auto zipped_order_line_kv = thrust::make_zip_iterator(
                    thrust::make_tuple(d_order_line_keys.begin(), d_order_line_values.begin()));
                order_line_index->insert(zipped_order_line_kv, zipped_order_line_kv + num_order_lines);
            }

            thrust::sequence(dp_order_free_rows, dp_order_free_rows + tpcc_config.orderTableSize(), num_orders);
            thrust::sequence(
                dp_new_order_free_rows, dp_new_order_free_rows + tpcc_config.newOrderTableSize(), num_new_orders);
            thrust::sequence(
                dp_order_line_free_rows, dp_order_line_free_rows + tpcc_config.orderLineTableSize(), num_order_lines);
        }

        logger.Info("Finished loading initial data");
        size_t free, total;
        gpu_err_check(cudaMemGetInfo(&free, &total));
        logger.Info("GPU memory usage: {} / {}", formatSizeBytes(total - free), formatSizeBytes(total));

    }

    void indexTxns(TxnArrayType &txn_array, TxnParamArrayType &index_array, uint32_t epoch_id)
    {
        if (txn_array.device != DeviceType::GPU || index_array.device != DeviceType::GPU)
        {
            throw std::runtime_error("TpccGpuIndex only supports GPU transaction array");
        }
        auto &logger = Logger::GetInstance();

        // Shift the shadow's trailing snapshots at the START of the
        // epoch so the rollback target stays f_{E-2} throughout this
        // entire epoch (indexTxns, execution, flush). Captures the
        // values that were durable at the end of epochs E-1 and E-2
        // before any of this epoch's inserts increment the cursors.
        shadow_.shiftSnapshotsAtEpochStart(new_order_free_start, order_free_start, order_line_free_start);

        constexpr uint32_t block_size = 512;
        const uint32_t txn_blocks = (tpcc_config.num_txns + block_size - 1) / block_size;
        auto blocks_for = [](uint32_t n) { return (n + 255u) / 256u; };
        const uint32_t no_init = tpcc_config.num_warehouses * 10u * 900u;
        const uint32_t o_init  = tpcc_config.num_warehouses * 10u * 3000u;
        const uint32_t ol_init = tpcc_config.num_warehouses * 10u * 3000u * 15u;
        const uint32_t no_old = new_order_free_start, o_old = order_free_start, ol_old = order_line_free_start;

        prepareTpccIndexKernel<<<txn_blocks, block_size>>>(
            GpuPackedTxnArray(txn_array), d_order_insert, d_new_order_insert, d_order_line_insert, tpcc_config.num_txns);
        gpu_err_check(cudaPeekAtLastError());

        // Valid inserts and their entries, in entry order (the minting order).
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_order_insert, d_order_valid_insert,
            d_order_num_insert, tpcc_config.num_txns,
            [] __device__(OrderKey::baseType val) { return val != static_cast<OrderKey::baseType>(-1); });
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, thrust::counting_iterator<uint32_t>(0), d_o_ins_entry,
            d_order_num_insert, tpcc_config.num_txns, epic::timeline::detail::EntryHasKey<OrderKey::baseType>{d_order_insert});
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_new_order_insert, d_new_order_valid_insert,
            d_new_order_num_insert, tpcc_config.num_txns,
            [] __device__(NewOrderKey::baseType val) { return val != static_cast<NewOrderKey::baseType>(-1); });
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, thrust::counting_iterator<uint32_t>(0), d_no_ins_entry,
            d_new_order_num_insert, tpcc_config.num_txns, epic::timeline::detail::EntryHasKey<NewOrderKey::baseType>{d_new_order_insert});
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_order_line_insert, d_order_line_valid_insert,
            d_order_line_num_insert, tpcc_config.num_txns * 15,
            [] __device__(OrderLineKey::baseType val) { return val != static_cast<OrderLineKey::baseType>(-1); });
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, thrust::counting_iterator<uint32_t>(0), d_ol_ins_entry,
            d_order_line_num_insert, tpcc_config.num_txns * 15, epic::timeline::detail::EntryHasKey<OrderLineKey::baseType>{d_order_line_insert});

        uint32_t num_orders_inserts, num_new_orders_inserts, num_order_lines_inserts;
        gpu_err_check(cudaMemcpy(&num_orders_inserts, d_order_num_insert, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpu_err_check(
            cudaMemcpy(&num_new_orders_inserts, d_new_order_num_insert, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpu_err_check(
            cudaMemcpy(&num_order_lines_inserts, d_order_line_num_insert, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        logger.Trace("Number of orders inserts: {}", num_orders_inserts);
        logger.Trace("Number of new orders inserts: {}", num_new_orders_inserts);
        logger.Trace("Number of order lines inserts: {}", num_order_lines_inserts);

        // Which tables take the timeline path this epoch. NewOrder always
        // does on Delivery-bearing hybrid runs (it has deletes); Order and
        // OrderLine fall back to it when the bulk insert rejects a key.
        const bool delete_path = tpcc_config.execution_mode == ExecMode::HYBRID_STAGING
            && tpcc_config.txn_mix.delivery > 0;
        no_general_ = false;
        o_general_ = false;
        ol_general_ = false;
#ifdef EGAD_VALIDATION
        // Validation hook: run every growing table through the timeline
        // path, so the workloads without duplicate keys exercise it and must
        // reproduce the fast path's results exactly.
        static const bool kForceGeneral = std::getenv("EPIC_TIMELINE_FORCE_GENERAL") != nullptr;
        if (kForceGeneral) { no_general_ = true; o_general_ = true; ol_general_ = true; }
#endif

        // --- fast paths: bulk inserts, rejection detection, fallback ---
        if (!o_general_ && num_orders_inserts > 0) {
            const std::size_t before = order_index->get_size();
            auto zipped = thrust::make_zip_iterator(thrust::make_tuple(dp_order_valid_insert, dp_order_free_rows + order_free_start));
            order_index->insert(zipped, zipped + num_orders_inserts);
            if (order_index->get_size() - before != num_orders_inserts) {
                logger.Info("Epoch {}: Order insert rejected, resolving through the timeline", epoch_id);
                undoAccepted(order_index, dp_order_valid_insert, num_orders_inserts, o_init + o_old);
                o_general_ = true;
            }
        }
        if (!no_general_ && num_new_orders_inserts > 0) {
            const std::size_t before = new_order_index->get_size();
            auto zipped = thrust::make_zip_iterator(thrust::make_tuple(dp_new_order_valid_insert, dp_new_order_free_rows + new_order_free_start));
            new_order_index->insert(zipped, zipped + num_new_orders_inserts);
            if (new_order_index->get_size() - before != num_new_orders_inserts) {
                logger.Info("Epoch {}: NewOrder insert rejected, resolving through the timeline", epoch_id);
                undoAccepted(new_order_index, dp_new_order_valid_insert, num_new_orders_inserts, no_init + no_old);
                no_general_ = true;
            }
        }
        if (!ol_general_ && num_order_lines_inserts > 0) {
            if (use_flat_ol_index_) {
                h_flat_rejected[0] = 0; h_flat_rejected[1] = 0;
                k_flat_ol_bulk_insert_checked<<<blocks_for(num_order_lines_inserts), 256>>>(
                    d_order_line_valid_insert, d_order_line_free_rows + order_line_free_start,
                    num_order_lines_inserts, index_device_view.order_line_flat_view, d_flat_rejected, d_flat_rejected + 1);
                gpu_err_check(cudaPeekAtLastError());
                gpu_err_check(cudaDeviceSynchronize());
                if (h_flat_rejected[1] != 0) {
                    throw std::runtime_error("OrderLine flat index: " + std::to_string(h_flat_rejected[1]) +
                        " order ids exceed the per-district stride (" + std::to_string(order_line_flat_max_o_) +
                        ") in epoch " + std::to_string(epoch_id) + "; the order insert pool is too small for this run");
                }
                if (h_flat_rejected[0] != 0) {
                    logger.Info("Epoch {}: OrderLine insert rejected, resolving through the timeline", epoch_id);
                    undoAcceptedFlatOL(num_order_lines_inserts, ol_init + ol_old);
                    ol_general_ = true;
                }
            } else {
                const std::size_t before = order_line_index->get_size();
                auto zipped = thrust::make_zip_iterator(thrust::make_tuple(dp_order_line_valid_insert, dp_order_line_free_rows + order_line_free_start));
                order_line_index->insert(zipped, zipped + num_order_lines_inserts);
                if (order_line_index->get_size() - before != num_order_lines_inserts) {
                    logger.Info("Epoch {}: OrderLine insert rejected, resolving through the timeline", epoch_id);
                    undoAccepted(order_line_index, dp_order_line_valid_insert, num_order_lines_inserts, ol_init + ol_old);
                    ol_general_ = true;
                }
            }
        }

        // --- timeline paths ---
        if (no_general_ || o_general_ || ol_general_) {
            if (no_general_) no_tl_.ensure(tpcc_config.num_txns * 11, tpcc_config.num_txns * 11);
            if (o_general_)  o_tl_.ensure(tpcc_config.num_txns, tpcc_config.num_txns);
            if (ol_general_) ol_tl_.ensure(tpcc_config.num_txns * 15, tpcc_config.num_txns * 15);
            k_fill_growing_timeline_inputs<GpuPackedTxnArray><<<txn_blocks, block_size>>>(
                GpuPackedTxnArray(txn_array), tpcc_config.num_txns,
                no_general_ ? no_tl_.insKeys() : nullptr, no_general_ ? no_tl_.delKeys() : nullptr, no_general_ ? no_tl_.pos() : nullptr,
                o_general_ ? o_tl_.insKeys() : nullptr, o_general_ ? o_tl_.pos() : nullptr,
                ol_general_ ? ol_tl_.insKeys() : nullptr, ol_general_ ? ol_tl_.pos() : nullptr);
            gpu_err_check(cudaPeekAtLastError());
            gpu_err_check(cudaStreamSynchronize(0));
            if (no_general_) {
                no_tl_.build([this](const NewOrderKey::baseType* keys, uint32_t n, uint32_t* out) {
                        thrust::device_ptr<const NewOrderKey::baseType> k(keys); thrust::device_ptr<uint32_t> o(out);
                        new_order_index->find(k, k + n, o);
                    }, no_init + no_old);
                num_new_orders_inserts = no_tl_.numMinted();
            }
            if (o_general_) {
                o_tl_.build([this](const OrderKey::baseType* keys, uint32_t n, uint32_t* out) {
                        thrust::device_ptr<const OrderKey::baseType> k(keys); thrust::device_ptr<uint32_t> o(out);
                        order_index->find(k, k + n, o);
                    }, o_init + o_old);
                num_orders_inserts = o_tl_.numMinted();
            }
            if (ol_general_) {
                ol_tl_.build([this, blocks_for](const OrderLineKey::baseType* keys, uint32_t n, uint32_t* out) {
                        if (use_flat_ol_index_) {
                            k_flat_ol_find_many<<<blocks_for(n), 256>>>(keys, n, index_device_view.order_line_flat_view, out);
                            gpu_err_check(cudaPeekAtLastError());
                            gpu_err_check(cudaStreamSynchronize(0));
                        } else {
                            thrust::device_ptr<const OrderLineKey::baseType> k(keys); thrust::device_ptr<uint32_t> o(out);
                            order_line_index->find(k, k + n, o);
                        }
                    }, ol_init + ol_old);
                num_order_lines_inserts = ol_tl_.numMinted();
            }
        }
        order_free_start += num_orders_inserts;
        new_order_free_start += num_new_orders_inserts;
        order_line_free_start += num_order_lines_inserts;
        logger.Trace("Order free rows used: {}", order_free_start);
        logger.Trace("New order free rows used: {}", new_order_free_start);
        logger.Trace("Order line free rows used: {}", order_line_free_start);

        // NewOrder fast delete path, pass 1: where each delivered record's
        // first delete sits, so the lookups below can hide the record from
        // the transactions ordered after it.
        if (delete_path && !no_general_) {
            if (!d_no_del_pos) {
                gpu_err_check(cudaMalloc(&d_no_del_pos, sizeof(uint32_t) * tpcc_config.newOrderTableSize()));
                gpu_err_check(cudaMemset(d_no_del_pos, 0xff, sizeof(uint32_t) * tpcc_config.newOrderTableSize()));
            }
        }

        // Per-table resolution state for the lookup kernel.
        index_device_view.o_tv = GrowingTableView<24>{};
        index_device_view.no_tv = GrowingTableView<24>{};
        index_device_view.ol_tv = GrowingTableView<22>{};
        index_device_view.o_tv.minted_begin = o_init + o_old;   index_device_view.o_tv.num_minted = num_orders_inserts;
        index_device_view.no_tv.minted_begin = no_init + no_old; index_device_view.no_tv.num_minted = num_new_orders_inserts;
        index_device_view.ol_tv.minted_begin = ol_init + ol_old; index_device_view.ol_tv.num_minted = num_order_lines_inserts;
        index_device_view.o_tv.ins_entry = o_general_ ? o_tl_.insEntries() : d_o_ins_entry;
        index_device_view.no_tv.ins_entry = no_general_ ? no_tl_.insEntries() : d_no_ins_entry;
        index_device_view.ol_tv.ins_entry = ol_general_ ? ol_tl_.insEntries() : d_ol_ins_entry;
        index_device_view.o_tv.entries_per_txn = 1;
        index_device_view.no_tv.entries_per_txn = no_general_ ? 11u : 1u;
        index_device_view.ol_tv.entries_per_txn = 15;
        if (o_general_)  index_device_view.o_tv.tl = o_tl_.view();
        if (no_general_) index_device_view.no_tv.tl = no_tl_.view();
        if (ol_general_) index_device_view.ol_tv.tl = ol_tl_.view();
        if (delete_path && !no_general_) {
            k_record_no_deletes<GpuPackedTxnArray><<<txn_blocks, block_size>>>(GpuPackedTxnArray(txn_array),
                index_device_view, tpcc_config.num_txns, d_no_del_pos);
            gpu_err_check(cudaPeekAtLastError());
            index_device_view.no_tv.del_pos = d_no_del_pos;
        }

        // === Per-epoch CPU shadow mirror ===
        // Gated on durable mode (EPIC_DURABLE_STORE / EPIC_RECOVER_FROM),
        // so throughput runs stay baseline-equivalent. When active, D2H the
        // per-table minted keys (entry order: the j-th key is CRID
        // num_initial + old_free_start + j) into the shadow's pinned host
        // buffers and delegate the host-side mirror update to
        // TpccCpuShadowIndex. The D2Hs are on the default stream so they
        // drain the GPU work issued above.
        static const bool kTpccShadowMirror = []{
            // Real-fault durable-recovery driver: the durable
            // worker must persist runtime insert keys into the durable shadow
            // arrays (via mirrorEpoch's memcpy), and the recover process keeps the
            // shadow consistent while it replays. Gated on the durable env, so
            // throughput runs stay baseline-equivalent.
            return std::getenv("EPIC_DURABLE_STORE") != nullptr ||
                   std::getenv("EPIC_RECOVER_FROM") != nullptr;
        }();
        if (kTpccShadowMirror) {
            if (num_orders_inserts > 0) {
                gpu_err_check(cudaMemcpy(shadow_.h_o_keys(), o_general_ ? o_tl_.mintedKeys() : d_order_valid_insert,
                    num_orders_inserts * sizeof(OrderKey::baseType), cudaMemcpyDeviceToHost));
            }
            if (num_new_orders_inserts > 0) {
                gpu_err_check(cudaMemcpy(shadow_.h_no_keys(), no_general_ ? no_tl_.mintedKeys() : d_new_order_valid_insert,
                    num_new_orders_inserts * sizeof(NewOrderKey::baseType), cudaMemcpyDeviceToHost));
            }
            if (num_order_lines_inserts > 0) {
                gpu_err_check(cudaMemcpy(shadow_.h_ol_keys(), ol_general_ ? ol_tl_.mintedKeys() : d_order_line_valid_insert,
                    num_order_lines_inserts * sizeof(OrderLineKey::baseType), cudaMemcpyDeviceToHost));
            }
            shadow_.mirrorEpoch(
                num_new_orders_inserts, num_orders_inserts, num_order_lines_inserts,
                /*no_old=*/no_old, /*o_old=*/o_old, /*ol_old=*/ol_old);
        }

        // TODO: revert hack
        indexTpccTxnKernel<<<txn_blocks, block_size>>>(
            GpuPackedTxnArray(txn_array), GpuTxnArrayType(index_array), index_device_view, tpcc_config.num_txns);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaDeviceSynchronize());

        // Epoch-boundary index updates of the timeline tables: erase the
        // keys that end absent or replaced, then insert the final lives
        // minted this epoch.
        if (o_general_) finalizeCuco(order_index, o_tl_);
        if (no_general_) finalizeCuco(new_order_index, no_tl_);
        if (ol_general_) {
            if (use_flat_ol_index_) {
                if (ol_tl_.numErase() > 0) {
                    throw std::runtime_error("OrderLine timeline produced an erase; the table has no deletes");
                }
                if (ol_tl_.numInsert() > 0) {
                    k_flat_ol_assign_many<<<blocks_for(ol_tl_.numInsert()), 256>>>(ol_tl_.insertKeys(), ol_tl_.insertCrids(),
                        ol_tl_.numInsert(), index_device_view.order_line_flat_view);
                    gpu_err_check(cudaPeekAtLastError());
                    gpu_err_check(cudaDeviceSynchronize());
                }
            } else {
                finalizeCuco(order_line_index, ol_tl_);
            }
        }

        // NewOrder deletes: the effective deletes' (key, CRID) pairs go to
        // the durable delete log; the CRID list stays on device for the NO
        // stager's reclaim-first marking. hybrid_staging only: the
        // cpu_only / gpu_only baselines stay pure Epic.
        num_no_deletes_this_epoch = 0;
        if (delete_path) {
            const NewOrderKey::baseType* edel_keys = nullptr;
            if (no_general_) {
                num_no_deletes_this_epoch = no_tl_.numEffectiveDeletes();
                edel_keys = no_tl_.edelKeys();
                d_no_edel_out = no_tl_.edelCrids();
            } else {
                // Fast delete path, pass 3: the effective deletes in slot order,
                // then the erase and the reset of del_pos for the records recorded.
                const size_t n_slots = static_cast<size_t>(tpcc_config.num_txns) * 10;
                k_no_effective_deletes<GpuPackedTxnArray, GpuTxnArrayType><<<txn_blocks, block_size>>>(
                    GpuPackedTxnArray(txn_array), GpuTxnArrayType(index_array), tpcc_config.num_txns, d_no_del_pos,
                    d_no_edel_key, d_no_edel_crid, d_no_edel_flag);
                gpu_err_check(cudaPeekAtLastError());
                {
                    size_t tmp = no_flagged_temp_bytes;
                    cub::DeviceSelect::Flagged(d_no_flagged_temp, tmp,
                        d_no_edel_key, d_no_edel_flag, d_no_valid_deletes, d_num_no_delete, n_slots);
                }
                {
                    size_t tmp = no_flagged_temp_bytes;
                    cub::DeviceSelect::Flagged(d_no_flagged_temp, tmp,
                        d_no_edel_crid, d_no_edel_flag, d_no_valid_delete_crids, d_num_no_delete, n_slots);
                }
                gpu_err_check(cudaStreamSynchronize(0));
                num_no_deletes_this_epoch = *h_num_no_delete;
                edel_keys = d_no_valid_deletes;
                d_no_edel_out = d_no_valid_delete_crids;
                if (num_no_deletes_this_epoch > 0) {
                    k_reset_del_pos<<<blocks_for(num_no_deletes_this_epoch), 256>>>(d_no_valid_delete_crids,
                        num_no_deletes_this_epoch, d_no_del_pos);
                    gpu_err_check(cudaPeekAtLastError());
                    new_order_index->erase(dp_no_valid_deletes, dp_no_valid_deletes + num_no_deletes_this_epoch);
                    gpu_err_check(cudaStreamSynchronize(0));
                }
            }
            logger.Info("Found {} new order deletes", num_no_deletes_this_epoch);
            if (num_no_deletes_this_epoch > 0) {
                gpu_err_check(cudaMemcpy(
                    shadow_.h_no_delete_keys(), edel_keys,
                    num_no_deletes_this_epoch * sizeof(NewOrderKey::baseType), cudaMemcpyDeviceToHost));
                gpu_err_check(cudaMemcpy(
                    shadow_.h_no_delete_crids(), d_no_edel_out,
                    num_no_deletes_this_epoch * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                shadow_.mirrorEpochNoDeletes(num_no_deletes_this_epoch, no_delete_count);
                no_delete_count += num_no_deletes_this_epoch;
            }
        }
        logger.Info("Finished indexing transactions");
    }

    // Fast-path undo: the bulk insert accepted an unknown subset of the
    // epoch's insert keys with their tentative CRIDs. A key whose lookup
    // returns a tentative CRID was accepted; erase those so the map is back
    // at its epoch-start state before the timeline resolves the epoch.
    template <typename MapT, typename KeyT>
    void undoAccepted(std::shared_ptr<MapT>& map, thrust::device_ptr<KeyT> keys, uint32_t n, uint32_t tentative_begin)
    {
        thrust::device_vector<uint32_t> found(n);
        thrust::device_vector<KeyT> erase(n);
        map->find(keys, keys + n, found.begin());
        auto is_tentative = [tentative_begin, n] __device__ (uint32_t v) { return v - tentative_begin < n; };
        auto end = thrust::copy_if(keys, keys + n, found.begin(), erase.begin(), is_tentative);
        const uint32_t n_erase = static_cast<uint32_t>(end - erase.begin());
        if (n_erase > 0) map->erase(erase.begin(), erase.begin() + n_erase);
        gpu_err_check(cudaStreamSynchronize(0));
    }
    void undoAcceptedFlatOL(uint32_t n, uint32_t tentative_begin)
    {
        thrust::device_vector<uint32_t> found(n);
        k_flat_ol_find_many<<<(n + 255u) / 256u, 256>>>(d_order_line_valid_insert, n,
            index_device_view.order_line_flat_view, thrust::raw_pointer_cast(found.data()));
        gpu_err_check(cudaPeekAtLastError());
        thrust::device_vector<uint32_t> sentinel(n, kAbsent);
        thrust::device_vector<OrderLineKey::baseType> keys(n);
        auto is_tentative = [tentative_begin, n] __device__ (uint32_t v) { return v - tentative_begin < n; };
        auto end = thrust::copy_if(dp_order_line_valid_insert, dp_order_line_valid_insert + n, found.begin(), keys.begin(), is_tentative);
        const uint32_t n_erase = static_cast<uint32_t>(end - keys.begin());
        if (n_erase > 0) {
            k_flat_ol_assign_many<<<(n_erase + 255u) / 256u, 256>>>(thrust::raw_pointer_cast(keys.data()),
                thrust::raw_pointer_cast(sentinel.data()), n_erase, index_device_view.order_line_flat_view);
            gpu_err_check(cudaPeekAtLastError());
        }
        gpu_err_check(cudaDeviceSynchronize());
    }
    template <typename MapT, typename TimelineT>
    void finalizeCuco(std::shared_ptr<MapT>& map, TimelineT& tl)
    {
        using KeyT = typename std::remove_const<typename std::remove_pointer<decltype(tl.eraseKeys())>::type>::type;
        if (tl.numErase() > 0) {
            thrust::device_ptr<const KeyT> k(tl.eraseKeys());
            map->erase(k, k + tl.numErase());
        }
        if (tl.numInsert() > 0) {
            thrust::device_ptr<const KeyT> k(tl.insertKeys());
            thrust::device_ptr<const uint32_t> v(tl.insertCrids());
            auto zipped = thrust::make_zip_iterator(thrust::make_tuple(k, v));
            map->insert(zipped, zipped + tl.numInsert());
        }
        gpu_err_check(cudaStreamSynchronize(0));
    }

    // === CPU-shadow recovery API ===
    //
    // rebuildIndexesFromShadow resets the per-table free_start cursors
    // and trailing snapshots, erases stragglers from the 3 growing
    // shadows, then re-uploads every non-sentinel shadow entry to the
    // GPU. The upload recreates the 8 active cuco maps (W/D/C/I/S/NO/O/OL,
    // History excluded), reseeds the 3 d_*_free_rows arrays, repopulates
    // the OL flat array when active, and rebuilds index_device_view.
    // Assumes the CUDA context is alive (d_*_free_rows and
    // d_order_line_flat_ are reused, not reallocated); in a fresh recover
    // process the constructor has already allocated them.

    TpccFreeStarts getInsertCounts() const
    {
        return TpccFreeStarts{new_order_free_start, order_free_start, order_line_free_start};
    }

    // === Shadow -> cuco upload helper (recovery rebuild) ===
    //
    // Walks the shadow vector, collects every non-sentinel (decoded_key,
    // expected_crid) pair, and batch-uploads them into the supplied cuco
    // map. Streams
    // in 1 M-entry batches to cap device transient memory (same pattern
    // as YCSB's rebuildCucoFromShadow).
    template <typename CucoT, typename DecodeFn>
    void uploadShadowToCuco_(const char* name, CucoT& cuco,
                             const std::vector<uint32_t>& shadow, DecodeFn decode)
    {
        using KeyBaseT = decltype(decode(uint32_t{}));
        auto& logger = Logger::GetInstance();

        std::vector<KeyBaseT> h_keys;
        std::vector<uint32_t> h_values;
        h_keys.reserve(shadow.size() / 2);
        h_values.reserve(shadow.size() / 2);
        for (size_t i = 0; i < shadow.size(); ++i) {
            if (shadow[i] == TpccCpuShadowIndex::kSentinel) continue;
            h_keys.push_back(decode(static_cast<uint32_t>(i)));
            h_values.push_back(shadow[i]);
        }
        if (h_keys.empty()) {
            logger.Info("[SHADOW-REBUILD] {} no entries to upload", name);
            return;
        }

        constexpr size_t kBatch = 1u << 20;
        const size_t cap = std::min(kBatch, h_keys.size());
        thrust::device_vector<KeyBaseT> d_keys(cap);
        thrust::device_vector<uint32_t> d_vals(cap);

        size_t uploaded = 0;
        while (uploaded < h_keys.size()) {
            const size_t take = std::min(kBatch, h_keys.size() - uploaded);
            thrust::copy(h_keys.begin()   + uploaded,
                         h_keys.begin()   + uploaded + take, d_keys.begin());
            thrust::copy(h_values.begin() + uploaded,
                         h_values.begin() + uploaded + take, d_vals.begin());
            auto zipped = thrust::make_zip_iterator(
                thrust::make_tuple(d_keys.begin(), d_vals.begin()));
            cuco.insert(zipped, zipped + take);
            uploaded += take;
        }

        logger.Info("[SHADOW-REBUILD] {} uploaded {} entries", name, h_keys.size());
    }

    void rebuildIndexesFromShadow(TpccFreeStarts current, uint32_t current_no_delete_count)
    {
        auto &logger = Logger::GetInstance();
        gpu_err_check(cudaStreamSynchronize(0));
        // Resync the NO delete-log cursor to the rollback point so
        // replay's re-applied deletes overwrite the log tail at the same
        // positions.
        no_delete_count = current_no_delete_count;
        num_no_deletes_this_epoch = 0;
        logger.Info("[SHADOW-REBUILD] starting: NO={} O={} OL={}",
                    current.new_order, current.order, current.order_line);

        // Sync the GPU-side free-row cursors and the shadow-side snapshots
        // to the rollback point. The shadow handles the snapshot reset
        // and the straggler erase (any shadow entry with CRID >= the
        // per-table max is a leftover from a not-yet-flushed epoch and
        // gets cleared so the upload below matches a fresh
        // end-of-(E-2) state).
        new_order_free_start  = current.new_order;
        order_free_start      = current.order;
        order_line_free_start = current.order_line;
        shadow_.syncSnapshotsToRollback(current);

        const uint32_t W = tpcc_config.num_warehouses;
        const uint32_t no_max = W * 10u * 900u        + current.new_order;
        const uint32_t o_max  = W * 10u * 3000u       + current.order;
        const uint32_t ol_max = W * 10u * 3000u * 15u + current.order_line;
        shadow_.eraseStragglers(no_max, o_max, ol_max);

        // === GPU re-upload ===
        // Tear down and recreate the 8 active cuco indices, reseed the
        // free-row arrays, repopulate the OL flat index (when active),
        // and batch-upload every non-sentinel shadow entry to its GPU
        // index. History is excluded -- it is never populated, has no
        // shadow, and has nothing to restore. Assumes the CUDA context
        // is alive (d_*_free_rows and d_order_line_flat_ are reused);
        // in a fresh recover process the constructor has already
        // allocated them.

        // 1. Recreate cuco maps at their original capacities, matching the
        //    ctor (TpccGpuIndexImpl::TpccGpuIndexImpl initializer list).
        warehouse_index = std::make_shared<WarehouseIndexType>(
            static_cast<size_t>(std::ceil(tpcc_config.warehouseTableSize() / load_factor)),
            warehouse_key_sentinel, value_sentinel);
        district_index = std::make_shared<DistrictIndexType>(
            static_cast<size_t>(std::ceil(tpcc_config.districtTableSize() / load_factor)),
            district_key_sentinel, value_sentinel);
        customer_index = std::make_shared<CustomerIndexType>(
            static_cast<size_t>(std::ceil(tpcc_config.customerTableSize() / load_factor)),
            customer_key_sentinel, value_sentinel);
        item_index = std::make_shared<ItemIndexType>(
            static_cast<size_t>(std::ceil(tpcc_config.itemTableSize() / load_factor)),
            item_key_sentinel, value_sentinel);
        stock_index = std::make_shared<StockIndexType>(
            static_cast<size_t>(std::ceil(tpcc_config.stockTableSize() / load_factor)),
            stock_key_sentinel, value_sentinel);
        new_order_index = std::make_shared<NewOrderIndexType>(
            static_cast<size_t>(std::ceil(tpcc_config.newOrderTableSize() / load_factor)),
            new_order_key_sentinel, value_sentinel, new_order_erased_sentinel);
        order_index = std::make_shared<OrderIndexType>(
            static_cast<size_t>(std::ceil(tpcc_config.orderTableSize() / load_factor)),
            order_key_sentinel, value_sentinel);
        order_line_index = std::make_shared<OrderLineIndexType>(
            cucoOLCapacity(tpcc_config), order_line_key_sentinel, value_sentinel);

        // 2. Reseed d_*_free_rows for the 3 growing tables. After this,
        //    free_rows[free_start + j] is the CRID the next runtime insert
        //    will get, matching loadInitialData()'s thrust::sequence.
        const uint32_t num_initial_no = tpcc_config.num_warehouses * 10u * 900u;
        const uint32_t num_initial_o  = tpcc_config.num_warehouses * 10u * 3000u;
        const uint32_t num_initial_ol = tpcc_config.num_warehouses * 10u * 3000u * 15u;
        thrust::sequence(dp_new_order_free_rows,
                         dp_new_order_free_rows + tpcc_config.newOrderTableSize(),
                         num_initial_no);
        thrust::sequence(dp_order_free_rows,
                         dp_order_free_rows + tpcc_config.orderTableSize(),
                         num_initial_o);
        thrust::sequence(dp_order_line_free_rows,
                         dp_order_line_free_rows + tpcc_config.orderLineTableSize(),
                         num_initial_ol);

        // 3. OL flat-index fast path. shadow_.shadow_ol() uses the same dense
        //    encoding and size as d_order_line_flat_, so a single H2D
        //    copy reconstructs the whole flat index (sentinel slots
        //    included). Skipped when flat is off.
        if (use_flat_ol_index_) {
            gpu_err_check(cudaMemcpy(d_order_line_flat_, shadow_.shadow_ol().data(),
                order_line_flat_size_ * sizeof(uint32_t),
                cudaMemcpyHostToDevice));
        }

        // 4. Batch-upload each cuco-indexed shadow. The decode lambdas
        //    define the dense-idx -> key encoding in one place.
        uploadShadowToCuco_("W", *warehouse_index, shadow_.shadow_w(),
            [](uint32_t i) -> WarehouseKey::baseType {
                return WarehouseKey{static_cast<WarehouseKey::baseType>(i + 1u)}.base_key;
            });
        uploadShadowToCuco_("D", *district_index, shadow_.shadow_d(),
            [](uint32_t i) -> DistrictKey::baseType {
                uint32_t w = i / 10u + 1u, d = i % 10u + 1u;
                return DistrictKey{static_cast<DistrictKey::baseType>(d),
                                   static_cast<DistrictKey::baseType>(w)}.base_key;
            });
        uploadShadowToCuco_("C", *customer_index, shadow_.shadow_c(),
            [](uint32_t i) -> CustomerKey::baseType {
                uint32_t w = i / 30000u + 1u;
                uint32_t d = (i / 3000u) % 10u + 1u;
                uint32_t c = i % 3000u + 1u;
                return CustomerKey{static_cast<CustomerKey::baseType>(c),
                                   static_cast<CustomerKey::baseType>(d),
                                   static_cast<CustomerKey::baseType>(w)}.base_key;
            });
        uploadShadowToCuco_("I", *item_index, shadow_.shadow_i(),
            [](uint32_t i) -> ItemKey::baseType {
                return ItemKey{static_cast<ItemKey::baseType>(i + 1u)}.base_key;
            });
        uploadShadowToCuco_("S", *stock_index, shadow_.shadow_s(),
            [](uint32_t i) -> StockKey::baseType {
                uint32_t w = i / 100000u + 1u;
                uint32_t s = i % 100000u + 1u;
                return StockKey{static_cast<StockKey::baseType>(s),
                                static_cast<StockKey::baseType>(w)}.base_key;
            });
        uploadShadowToCuco_("NO", *new_order_index, shadow_.shadow_no(),
            [maxO_no = shadow_.max_o_orders()](uint32_t i) -> NewOrderKey::baseType {
                uint32_t per_w = 10u * maxO_no;
                uint32_t w = i / per_w + 1u;
                uint32_t r1 = i % per_w;
                uint32_t d = r1 / maxO_no + 1u;
                uint32_t o = r1 % maxO_no + 1u;
                return NewOrderKey{static_cast<NewOrderKey::baseType>(o),
                                   static_cast<NewOrderKey::baseType>(d),
                                   static_cast<NewOrderKey::baseType>(w)}.base_key;
            });
        uploadShadowToCuco_("O", *order_index, shadow_.shadow_o(),
            [maxO_no = shadow_.max_o_orders()](uint32_t i) -> OrderKey::baseType {
                uint32_t per_w = 10u * maxO_no;
                uint32_t w = i / per_w + 1u;
                uint32_t r1 = i % per_w;
                uint32_t d = r1 / maxO_no + 1u;
                uint32_t o = r1 % maxO_no + 1u;
                return OrderKey{static_cast<OrderKey::baseType>(o),
                                static_cast<OrderKey::baseType>(d),
                                static_cast<OrderKey::baseType>(w)}.base_key;
            });
        if (!use_flat_ol_index_) {
            const uint32_t maxO_cap = 3000u;
            uploadShadowToCuco_("OL", *order_line_index, shadow_.shadow_ol(),
                [maxO_cap](uint32_t idx) -> OrderLineKey::baseType {
                    uint32_t per_wd = maxO_cap * 15u;
                    uint32_t per_w  = 10u * per_wd;
                    uint32_t w  = idx / per_w + 1u;
                    uint32_t r1 = idx % per_w;
                    uint32_t d  = r1 / per_wd + 1u;
                    uint32_t r2 = r1 % per_wd;
                    uint32_t o  = r2 / 15u + 1u;
                    uint32_t ol = r2 % 15u + 1u;
                    return OrderLineKey{static_cast<OrderLineKey::baseType>(o),
                                        static_cast<OrderLineKey::baseType>(d),
                                        static_cast<OrderLineKey::baseType>(w),
                                        static_cast<OrderLineKey::baseType>(ol)}.base_key;
                });
        }

        // 5. Rebuild index_device_view to point at the fresh cuco device
        //    views. Kernels read it by value each epoch, so the next
        //    indexTxns() picks up the new pointers automatically. Same
        //    9-arg aggregate init as the ctor; trailing flat-OL fields
        //    default to use_flat_ol=false / nullptr flat view.
        index_device_view = tpccGpuIndexFindView{
            warehouse_index->get_device_view(),
            district_index->get_device_view(),
            customer_index->get_device_view(),
            history_index->get_device_view(),
            new_order_index->get_device_view(),
            order_index->get_device_view(),
            order_line_index->get_device_view(),
            item_index->get_device_view(),
            stock_index->get_device_view(),
        };
        if (use_flat_ol_index_) {
            index_device_view.use_flat_ol = true;
            index_device_view.order_line_flat_view =
                OrderLineFlatView{d_order_line_flat_, order_line_flat_max_o_};
        }

        gpu_err_check(cudaStreamSynchronize(0));
        logger.Info("[SHADOW-REBUILD] GPU re-upload complete");
    }
};

template <typename TxnArrayType, typename TxnParamArrayType>
TpccGpuIndex<TxnArrayType, TxnParamArrayType>::TpccGpuIndex(TpccConfig tpcc_config, TpccCpuShadowIndex& shadow)
    : tpcc_config(tpcc_config)
{
    gpu_index_impl = std::make_any<TpccGpuIndexImpl<TxnArrayType, TxnParamArrayType, TpccGpuTxnArrayT>>(tpcc_config, shadow);
}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuIndex<TxnArrayType, TxnParamArrayType>::loadInitialData()
{
    auto &impl = std::any_cast<TpccGpuIndexImpl<TxnArrayType, TxnParamArrayType, TpccGpuTxnArrayT> &>(gpu_index_impl);
    impl.loadInitialData();
}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuIndex<TxnArrayType, TxnParamArrayType>::indexTxns(
    TxnArrayType &txn_array, TxnParamArrayType &index_array, uint32_t epoch_id)
{
    auto &impl = std::any_cast<TpccGpuIndexImpl<TxnArrayType, TxnParamArrayType, TpccGpuTxnArrayT> &>(gpu_index_impl);
    impl.indexTxns(txn_array, index_array, epoch_id);
}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuIndex<TxnArrayType, TxnParamArrayType>::rebuildIndexesFromShadow(TpccFreeStarts current_free_starts,
    uint32_t current_no_delete_count)
{
    auto &impl = std::any_cast<TpccGpuIndexImpl<TxnArrayType, TxnParamArrayType, TpccGpuTxnArrayT> &>(gpu_index_impl);
    impl.rebuildIndexesFromShadow(current_free_starts, current_no_delete_count);
}

template <typename TxnArrayType, typename TxnParamArrayType>
TpccFreeStarts TpccGpuIndex<TxnArrayType, TxnParamArrayType>::getInsertCounts() const
{
    auto const &impl = std::any_cast<TpccGpuIndexImpl<TxnArrayType, TxnParamArrayType, TpccGpuTxnArrayT> const &>(gpu_index_impl);
    return impl.getInsertCounts();
}

template <typename TxnArrayType, typename TxnParamArrayType>
uint32_t TpccGpuIndex<TxnArrayType, TxnParamArrayType>::getNoDeleteCount() const
{
    auto const &impl = std::any_cast<TpccGpuIndexImpl<TxnArrayType, TxnParamArrayType, TpccGpuTxnArrayT> const &>(gpu_index_impl);
    return impl.no_delete_count;
}

template <typename TxnArrayType, typename TxnParamArrayType>
const uint32_t* TpccGpuIndex<TxnArrayType, TxnParamArrayType>::noDeleteCridsDevice() const
{
    auto const &impl = std::any_cast<TpccGpuIndexImpl<TxnArrayType, TxnParamArrayType, TpccGpuTxnArrayT> const &>(gpu_index_impl);
    return impl.d_no_edel_out;
}

template <typename TxnArrayType, typename TxnParamArrayType>
uint32_t TpccGpuIndex<TxnArrayType, TxnParamArrayType>::numNoDeletesThisEpoch() const
{
    auto const &impl = std::any_cast<TpccGpuIndexImpl<TxnArrayType, TxnParamArrayType, TpccGpuTxnArrayT> const &>(gpu_index_impl);
    return impl.num_no_deletes_this_epoch;
}

#ifdef EGAD_VALIDATION
template <typename TxnArrayType, typename TxnParamArrayType>
uint32_t TpccGpuIndex<TxnArrayType, TxnParamArrayType>::verifyNoLiveMapping(
    const std::vector<std::pair<NewOrderKey::baseType, uint32_t>>& live,
    const std::vector<NewOrderKey::baseType>& dead) const
{
    auto const &impl = std::any_cast<TpccGpuIndexImpl<TxnArrayType, TxnParamArrayType, TpccGpuTxnArrayT> const &>(gpu_index_impl);
    auto &logger = Logger::GetInstance();
    uint32_t mismatches = 0, dead_found = 0;
    constexpr size_t kBatch = 1u << 20;
    thrust::device_vector<NewOrderKey::baseType> d_keys(kBatch);
    thrust::device_vector<uint32_t> d_out(kBatch);
    std::vector<NewOrderKey::baseType> h_keys(kBatch);
    std::vector<uint32_t> h_out(kBatch);
    for (size_t off = 0; off < live.size(); off += kBatch) {
        const size_t n = std::min(kBatch, live.size() - off);
        for (size_t i = 0; i < n; ++i) h_keys[i] = live[off + i].first;
        thrust::copy(h_keys.begin(), h_keys.begin() + n, d_keys.begin());
        impl.new_order_index->find(d_keys.begin(), d_keys.begin() + n, d_out.begin());
        thrust::copy(d_out.begin(), d_out.begin() + n, h_out.begin());
        for (size_t i = 0; i < n; ++i) {
            if (h_out[i] != live[off + i].second) {
                if (mismatches < 10) {
                    logger.Error("[MAP-CHECK] NO key {} expected {} found {}", static_cast<uint64_t>(live[off + i].first),
                                 live[off + i].second, static_cast<int64_t>(h_out[i] == kAbsent ? -1 : static_cast<int64_t>(h_out[i])));
                }
                ++mismatches;
            }
        }
    }
    for (size_t off = 0; off < dead.size(); off += kBatch) {
        const size_t n = std::min(kBatch, dead.size() - off);
        thrust::copy(dead.begin() + off, dead.begin() + off + n, d_keys.begin());
        impl.new_order_index->find(d_keys.begin(), d_keys.begin() + n, d_out.begin());
        thrust::copy(d_out.begin(), d_out.begin() + n, h_out.begin());
        for (size_t i = 0; i < n; ++i) {
            if (h_out[i] != kAbsent) {
                if (dead_found < 10) logger.Error("[MAP-CHECK] delivered NO key {} still maps to {}", static_cast<uint64_t>(dead[off + i]), h_out[i]);
                ++dead_found;
            }
        }
    }
    const uint32_t violations = mismatches + dead_found;
    if (violations == 0) {
        logger.Info("[MAP-CHECK] PASS tpcc NO live={} dead_probed={} (index agrees with the log replay)", live.size(), dead.size());
    } else {
        logger.Error("[MAP-CHECK] FAILED tpcc NO live={} mismatches={} dead_probed={} dead_found={}",
                     live.size(), mismatches, dead.size(), dead_found);
    }
    return violations;
}
#endif // EGAD_VALIDATION

template class TpccGpuIndex<TpccTxnArrayT, TpccTxnParamArrayT>;
// template class TpccGpuIndex<TxnArray<TpccTxn>, TpccTxnParamArrayT>;

} // namespace epic::tpcc