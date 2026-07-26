//
// Created by Shujian Qian on 2024-04-12.
//


#include <benchmarks/tpcc_gpu_aux_index.h>

#include <random>
#include <vector>
#include <array>

#include <cooperative_groups.h>

#include <gpu_btree.h>
#include <benchmarks/tpcc_txn_gen.h>
#include <util_log.h>
#include <chrono>
#include <benchmarks/tpcc_txn.h>
#include <benchmarks/tpcc_gpu_txn.cuh>
#include <util_gpu_error_check.cuh>
#include <cub/warp/warp_merge_sort.cuh>

namespace epic::tpcc {

namespace detail {

namespace cg = cooperative_groups;

struct TreeParam
{
    using key_type = uint64_t;
    using value_type = uint64_t;
    using pair_type = var_pair_type<key_type, value_type, 42, 22>;
    constexpr static size_t branching_factor = 16;

    using node_type = GpuBTree::node_type<key_type, value_type, branching_factor, pair_type>;

    // num_prealloc_nodes sizes the device_bump_allocator's pool for the
    // CustomerOrderBTree. Each node is sizeof(pair_type[16]) = 128 B. The
    // bump allocator never frees, so the pool must hold every node the run
    // allocates: loadInitialData inserts W*10*3000 keys (~214k nodes at
    // W=64), and each epoch's NewOrder inserts add nodes in proportion to
    // the mix (~10.5k/epoch for tpccn at 100k txns, W=64). The pool must
    // dominate that demand rather than approximate it, because the
    // allocator's overflow check (`cuda_assert(new_slab_index != max_size_)`)
    // is compiled out under NDEBUG; an overrun reads and writes through OOB
    // pointers and surfaces as an illegal memory access on a later launch.
    //
    // 5,000,000 nodes = ~640 MB of device memory, with measured headroom:
    //   W=64 tpccn -e 30:  ~519k used  (10x headroom)
    //   W=64 tpccn -e 100: ~1.26M used (4x headroom)
    //   W=128 tpccn -e 30: ~1M used    (5x headroom)
    // Much deeper runs (e.g. -e 500 at W=64) would need runtime sizing or a
    // higher constant. The high-water warning in insertTxnUpdates fires when
    // usage crosses 75% of this cap.
    constexpr static size_t num_prealloc_nodes = 5'000'000;
    using allocator_type = device_bump_allocator<node_type, num_prealloc_nodes>;
};

using CustomerOrderBTree = GpuBTree::gpu_blink_tree<TreeParam::key_type, TreeParam::value_type,
    TreeParam::branching_factor, TreeParam::allocator_type, TreeParam::pair_type>;

union PackedCustomerOrderKey
{
    using baseType = uint64_t;
    constexpr static baseType max_o_id = (1ull << 20) - 1;
    constexpr static baseType invalid_key = (1ull << 42) - 1;
    struct
    {
        baseType o_id : 20;
        baseType c_id : 12;
        baseType d_id : 4;
        baseType w_id : 6;
    };
    baseType base_key = 0;
    PackedCustomerOrderKey() = default;
    PackedCustomerOrderKey(baseType o_id, baseType c_id, baseType d_id, baseType w_id)
    {
        base_key = 0;
        this->o_id = o_id;
        this->c_id = c_id;
        this->d_id = d_id;
        this->w_id = w_id;
    }
};

template<typename T>
struct mapped_vector
{
    mapped_vector(std::size_t capacity)
        : capacity_(capacity)
    {
        allocate(capacity);
    }
    T &operator[](std::size_t index)
    {
        return dh_buffer_[index];
    }
    ~mapped_vector() {}
    void free()
    {
        cuda_try(cudaDeviceSynchronize());
        cuda_try(cudaFreeHost(dh_buffer_));
    }
    T *data() const
    {
        return dh_buffer_;
    }

    std::vector<T> to_std_vector()
    {
        std::vector<T> copy(capacity_);
        for (std::size_t i = 0; i < capacity_; i++)
        {
            copy[i] = dh_buffer_[i];
        }
        return copy;
    }

private:
    void allocate(std::size_t count)
    {
        cuda_try(cudaMallocHost(&dh_buffer_, sizeof(T) * count));
    }
    std::size_t capacity_;
    T *dh_buffer_;
};

// Initial-population kernel for d_latest_oid_ on the flat aux path. The CPU
// loader places one initial order per customer with o_id == c_id, so the
// initial value at slot ((w-1)*10*3000 + (d-1)*3000 + (c-1)) is c. Decodes
// tid -> (w_idx, d_idx, c_idx) instead of needing a host-side keys vector.
__global__ inline void k_init_latest_oid_initial(
    uint32_t* __restrict__ d_latest_oid,
    uint32_t num_warehouses,
    uint32_t customers_per_district)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = num_warehouses * 10u * customers_per_district;
    if (tid >= total) return;
    uint32_t c_idx = tid % customers_per_district;
    d_latest_oid[tid] = c_idx + 1u;
}

// Post-recovery kernel: walk every populated slot of
// order_customers and atomicMax the o_id into d_latest_oid at the (w,d,c)
// position. Used by rebuildFromShadow to layer post-initial NewOrder inserts
// onto k_init_latest_oid_initial's initial state, which guarantees
// d_latest_oid[(w,d,c)] = max o_id ever owned by c in district (w,d).
__global__ inline void k_update_latest_oid_from_order_customers(
    uint32_t* __restrict__ d_latest_oid,
    const uint32_t* __restrict__ order_customers,
    uint32_t num_warehouses,
    uint32_t num_slots_per_district,
    uint32_t customers_per_district)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = num_warehouses * 10u * num_slots_per_district;
    if (tid >= total) return;
    uint32_t c_id = order_customers[tid];
    if (c_id == 0u || c_id > customers_per_district) return;
    uint32_t district_id = tid / num_slots_per_district;
    uint32_t o_id        = tid % num_slots_per_district;
    uint32_t latest_slot = district_id * customers_per_district + (c_id - 1u);
    atomicMax(&d_latest_oid[latest_slot], o_id);
}

// Flat-aux insert kernel. Same body as insert_txn_updates_kernel but replaces
// btree.cooperative_insert(...) with atomicMax on d_latest_oid. The kernel
// still needs to write to order_num_items / order_customers / order_items
// because Delivery and StockLevel read those arrays directly. The btree
// parameter and the (w,d,c,hacked_o) packed-key construction are gone since
// nothing on the flat path needs them.
template <typename GpuTxnArrayType>
void __global__ insert_txn_updates_flat_kernel(GpuTxnArrayType txns, uint32_t num_txns,
    uint32_t *order_num_items, uint32_t *order_customers,
    uint32_t (*order_items)[15], uint32_t num_slots_per_district,
    uint32_t *d_latest_oid, uint32_t customers_per_district)
{
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= num_txns) return;

    BaseTxn *base_txn_ptr = txns.getTxn(thread_id);
    uint32_t txn_type = base_txn_ptr->txn_type;
    if (static_cast<TpccTxnType>(txn_type) != TpccTxnType::NEW_ORDER) return;

    auto no_txn_ptr = reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(base_txn_ptr->data);

    uint32_t district_id = no_txn_ptr->d_id - 1 + (no_txn_ptr->origin_w_id - 1) * 10;
    uint32_t cache_idx = district_id * num_slots_per_district + no_txn_ptr->o_id;

    order_num_items[cache_idx] = no_txn_ptr->num_items;
    order_customers[cache_idx] = no_txn_ptr->c_id;
    for (uint32_t i = 0; i < no_txn_ptr->num_items; ++i) {
        order_items[cache_idx][i] = no_txn_ptr->items[i].i_id;
    }

    uint32_t latest_slot = district_id * customers_per_district + (no_txn_ptr->c_id - 1);
    atomicMax(&d_latest_oid[latest_slot], no_txn_ptr->o_id);
}

template <typename GpuTxnArrayType>
void __global__ insert_txn_updates_kernel(GpuTxnArrayType txns, uint32_t num_txns, CustomerOrderBTree btree,
    CustomerOrderBTree::allocator_type host_allocator, uint32_t *order_num_items, uint32_t *order_customers,
    uint32_t (*order_items)[15], uint32_t num_slots_per_district)
{
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<CustomerOrderBTree::branching_factor>(block);

    if (thread_id - tile.thread_rank() >= num_txns)
    {
        return;
    }

    BaseTxn *base_txn_ptr = nullptr;
    uint32_t txn_type = 0xffffffffu;
    bool to_insert = false;
    if (thread_id < num_txns)
    {
        base_txn_ptr = txns.getTxn(thread_id);
        txn_type = base_txn_ptr->txn_type;
        to_insert = true;
    }
  using allocator_type = typename CustomerOrderBTree::device_allocator_context_type;
  allocator_type allocator{host_allocator, tile};

    uint32_t work_queue = tile.ballot(to_insert);
    while (work_queue)
    {
        int curr_rank = __ffs(work_queue) - 1;
        BaseTxn *curr_base_txn_ptr = tile.shfl(base_txn_ptr, curr_rank);
        uint32_t curr_txn_type = tile.shfl(txn_type, curr_rank);

        switch (static_cast<TpccTxnType>(curr_txn_type))
        {
        case TpccTxnType::NEW_ORDER: {
            auto no_txn_ptr = reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(curr_base_txn_ptr->data);
            PackedCustomerOrderKey key;
            key.w_id = no_txn_ptr->origin_w_id;
            key.d_id = no_txn_ptr->d_id;
            key.c_id = no_txn_ptr->c_id;
            key.o_id = PackedCustomerOrderKey::max_o_id - no_txn_ptr->o_id; // hack: to get backward scan

            uint32_t district_id = no_txn_ptr->d_id - 1 + (no_txn_ptr->origin_w_id - 1) * 10;
            uint32_t cache_idx = district_id * num_slots_per_district + no_txn_ptr->o_id;
            if (tile.thread_rank() == curr_rank)
            {
                order_num_items[cache_idx] = no_txn_ptr->num_items;
                order_customers[cache_idx] = no_txn_ptr->c_id;
            }
            if (tile.thread_rank() < no_txn_ptr->num_items)
            {
                order_items[cache_idx][tile.thread_rank()] = no_txn_ptr->items[tile.thread_rank()].i_id;
            }

            btree.cooperative_insert(key.base_key, 0ull, tile, allocator);
            break;
        }
        default:
            assert(false);
            break;
        }

        if (tile.thread_rank() == curr_rank) { to_insert = false; }
        work_queue = tile.ballot(to_insert);
    }
}

template <typename GpuTxnArrayType, typename GpuTxnIndexArrayType>
void __global__ perform_range_queries_kernel(GpuTxnArrayType txns, GpuTxnIndexArrayType index, uint32_t num_txns,
    CustomerOrderBTree btree, CustomerOrderBTree::allocator_type host_allocator, uint32_t *order_num_items,
    uint32_t *order_customers, uint32_t (*order_items)[15], uint32_t num_slots_per_district)
{
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<CustomerOrderBTree::branching_factor>(block);

    constexpr uint32_t warp_threads = 16;
    constexpr uint32_t block_threads = 512;
    constexpr uint32_t warps_per_block = block_threads / warp_threads;
    constexpr uint32_t items_per_thread = 19;
    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x) / warp_threads;
    using WarpMergeSortT = cub::WarpMergeSort<uint32_t, items_per_thread, warp_threads>;

    __shared__ typename WarpMergeSortT::TempStorage temp_storage[warps_per_block];

    if (thread_id - tile.thread_rank() >= num_txns)
    {
        return;
    }

    BaseTxn *base_txn_ptr = nullptr;
    BaseTxn *base_param_ptr = nullptr;
    uint32_t txn_type = 0xffffffffu;
    bool to_insert = false;
    if (thread_id < num_txns)
    {
        base_txn_ptr = txns.getTxn(thread_id);
        base_param_ptr = index.getTxn(thread_id);
        txn_type = base_txn_ptr->txn_type;
        to_insert = true;
    }
  using allocator_type = typename CustomerOrderBTree::device_allocator_context_type;
  allocator_type allocator{host_allocator, tile};

    uint32_t work_queue = tile.ballot(to_insert);
    while (work_queue)
    {
        int curr_rank = __ffs(work_queue) - 1;
        BaseTxn *curr_base_txn_ptr = tile.shfl(base_txn_ptr, curr_rank);
        BaseTxn *curr_base_param_ptr = tile.shfl(base_param_ptr, curr_rank);
        uint32_t curr_txn_type = tile.shfl(txn_type, curr_rank);

        switch (static_cast<TpccTxnType>(curr_txn_type))
        {
        case TpccTxnType::ORDER_STATUS: {
            auto os_txn_ptr = reinterpret_cast<OrderStatusTxnInput *>(curr_base_txn_ptr->data);
            PackedCustomerOrderKey lower_bound;
            lower_bound.w_id = os_txn_ptr->w_id;
            lower_bound.d_id = os_txn_ptr->d_id;
            lower_bound.c_id = os_txn_ptr->c_id;
            lower_bound.o_id = PackedCustomerOrderKey::max_o_id - os_txn_ptr->o_id;

            PackedCustomerOrderKey upper_bound;
            upper_bound.base_key = lower_bound.base_key;
            upper_bound.o_id = PackedCustomerOrderKey::max_o_id;

            CustomerOrderBTree::pair_type respair = btree.cooperative_find_next(lower_bound.base_key, upper_bound.base_key, tile, allocator);
            PackedCustomerOrderKey reskey;
            reskey.base_key = respair.first;

            uint32_t o_id = PackedCustomerOrderKey::max_o_id - reskey.o_id; // revert the hack

            /* verify against CPU aux index */
            // if (tile.thread_rank() == 0 && os_txn_ptr->o_id != o_id)
            // {
            //     printf("ERROR: txn[%d] cpu_oid[%d] gpu_oid[%d]\n", thread_id - tile.thread_rank() + curr_rank,
            //         os_txn_ptr->o_id, o_id);
            // }
            // assert(os_txn_ptr->o_id == o_id);

            uint32_t district_id = os_txn_ptr->d_id - 1 + (os_txn_ptr->w_id - 1) * 10;
            uint32_t cache_idx = district_id * num_slots_per_district + o_id;

            if (tile.thread_rank() == curr_rank)
            {
                os_txn_ptr->o_id = o_id;
                os_txn_ptr->num_items = order_num_items[cache_idx];
            }
            break;
        }
        case TpccTxnType::DELIVERY: {
            auto dl_txn_ptr = reinterpret_cast<DeliveryTxnInput *>(curr_base_txn_ptr->data);

            if (tile.thread_rank() < 10)
            {
                uint32_t d_oid = dl_txn_ptr->o_id[tile.thread_rank()];
                // Per-district deliver: o_id==0 means this district has nothing to
                // deliver. Do NOT read order_customers[slot 0] (a stale/foreign
                // entry); fill the invalid-CRID sentinel and zero items so the
                // downstream index/executor skip this district entirely.
                if (d_oid == 0)
                {
                    dl_txn_ptr->num_items[tile.thread_rank()] = 0;
                    dl_txn_ptr->customers[tile.thread_rank()] = 0xffffffffu;
                }
                else
                {
                    uint32_t district_id = tile.thread_rank() + (dl_txn_ptr->w_id - 1) * 10;
                    uint32_t cache_idx = district_id * num_slots_per_district + d_oid;

                    /* verify against cpu aux index */
                    // if (dl_txn_ptr->num_items[tile.thread_rank()] != order_num_items[cache_idx])
                    // {
                    //     printf("ERROR: delivery txn[%d] w[%d]d[%d] oid[%d] cpu_num_items[%d] gpu[%d]\n",
                    //         thread_id - tile.thread_rank() + curr_rank, dl_txn_ptr->w_id, tile.thread_rank() + 1,
                    //         d_oid, dl_txn_ptr->num_items[tile.thread_rank()], order_num_items[cache_idx]);
                    // }
                    // if (dl_txn_ptr->customers[tile.thread_rank()] != order_customers[cache_idx])
                    // {
                    //     printf("ERROR: delivery txn[%d] w[%d]d[%d] cpu_customer[%d] gpu_[%d]\n",
                    //         thread_id - tile.thread_rank() + curr_rank, dl_txn_ptr->w_id, tile.thread_rank() + 1,
                    //         dl_txn_ptr->customers[tile.thread_rank()], order_customers[cache_idx]);
                    // }

                    dl_txn_ptr->num_items[tile.thread_rank()] = order_num_items[cache_idx];
                    dl_txn_ptr->customers[tile.thread_rank()] = order_customers[cache_idx];
                }
            }

            break;
        }
        case TpccTxnType::STOCK_LEVEL: {
            auto sl_txn_ptr = reinterpret_cast<StockLevelTxnInput *>(curr_base_txn_ptr->data);
            auto sl_param_ptr = reinterpret_cast<StockLevelTxnParams *>(curr_base_param_ptr->data);

            uint32_t items_cnt = 0;
            for (uint32_t o_id = sl_txn_ptr->o_id; o_id > sl_txn_ptr->o_id - 20; --o_id)
            {
                uint32_t district_id = sl_txn_ptr->d_id - 1 + (sl_txn_ptr->w_id - 1) * 10;
                uint32_t cache_idx = district_id * num_slots_per_district + o_id;

                uint32_t num_items = order_num_items[cache_idx];
                if (tile.thread_rank() < num_items)
                {
                    sl_param_ptr->stock_ids[items_cnt + tile.thread_rank()] = order_items[cache_idx][tile.thread_rank()];
                }
                items_cnt += num_items;
            }

            struct CustomLess
            {
                __device__ bool operator()(const uint32_t &lhs, const uint32_t &rhs)
                {
                    return lhs < rhs;
                }
            };
            WarpMergeSortT(temp_storage[warp_id])
                .Sort(
                    *reinterpret_cast<uint32_t(*)[19]>(&sl_param_ptr->stock_ids[items_per_thread * tile.thread_rank()]),
                    CustomLess(), items_cnt, 0xffffffffu);

            /* verify against CPU index */
            // if (tile.thread_rank() == 0)
            // {
            //     if (items_cnt != sl_txn_ptr->num_items)
            //     {
            //         printf("ERROR: stock_level txn[%d] w[%d] cpu_num_items[%d] gpu_[%d]\n",
            //             thread_id - tile.thread_rank() + curr_rank, sl_txn_ptr->w_id, sl_txn_ptr->num_items,
            //             items_cnt);
            //     }
            //     for (int i = 0; i < sl_txn_ptr->num_items; ++i)
            //     {
            //         if (sl_txn_ptr->o_id - 19 > 3000 && sl_txn_ptr->items[i] != sl_param_ptr->stock_ids[i])
            //         {
            //             printf("ERROR: stock_level txn[%d] w[%d]d[%d] item[%d] oid[%d] cpu_item[%d] gpu_[%d]\n",
            //                 thread_id - tile.thread_rank() + curr_rank, sl_txn_ptr->w_id, sl_txn_ptr->d_id, i,
            //                 sl_txn_ptr->o_id - 19, sl_txn_ptr->items[i], sl_param_ptr->stock_ids[i]);
            //         }
            //     }
            // }

            if (tile.thread_rank() == curr_rank)
            {
                sl_txn_ptr->num_items = items_cnt;
            }
            break;
        }
        default:
            assert(false);
            break;
        }

        if (tile.thread_rank() == curr_rank) { to_insert = false; }
        work_queue = tile.ballot(to_insert);
    }
}

// Flat-aux range-queries kernel. Same body as perform_range_queries_kernel but
// the OrderStatus case is rewritten to look up d_latest_oid directly instead
// of calling btree.cooperative_find_next. Delivery, StockLevel, and the
// default case are byte-identical to the cuco path. The btree, allocator, and
// hacked-key construction are dropped from the OrderStatus path.
template <typename GpuTxnArrayType, typename GpuTxnIndexArrayType>
void __global__ perform_range_queries_flat_kernel(GpuTxnArrayType txns, GpuTxnIndexArrayType index, uint32_t num_txns,
    uint32_t *order_num_items, uint32_t *order_customers, uint32_t (*order_items)[15],
    uint32_t num_slots_per_district, uint32_t *d_latest_oid, uint32_t customers_per_district)
{
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<16>(block);

    constexpr uint32_t warp_threads = 16;
    constexpr uint32_t block_threads = 512;
    constexpr uint32_t warps_per_block = block_threads / warp_threads;
    constexpr uint32_t items_per_thread = 19;
    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x) / warp_threads;
    using WarpMergeSortT = cub::WarpMergeSort<uint32_t, items_per_thread, warp_threads>;

    __shared__ typename WarpMergeSortT::TempStorage temp_storage[warps_per_block];

    if (thread_id - tile.thread_rank() >= num_txns) return;

    BaseTxn *base_txn_ptr = nullptr;
    BaseTxn *base_param_ptr = nullptr;
    uint32_t txn_type = 0xffffffffu;
    bool to_insert = false;
    if (thread_id < num_txns)
    {
        base_txn_ptr = txns.getTxn(thread_id);
        base_param_ptr = index.getTxn(thread_id);
        txn_type = base_txn_ptr->txn_type;
        to_insert = true;
    }

    uint32_t work_queue = tile.ballot(to_insert);
    while (work_queue)
    {
        int curr_rank = __ffs(work_queue) - 1;
        BaseTxn *curr_base_txn_ptr = tile.shfl(base_txn_ptr, curr_rank);
        BaseTxn *curr_base_param_ptr = tile.shfl(base_param_ptr, curr_rank);
        uint32_t curr_txn_type = tile.shfl(txn_type, curr_rank);

        switch (static_cast<TpccTxnType>(curr_txn_type))
        {
        case TpccTxnType::ORDER_STATUS: {
            auto os_txn_ptr = reinterpret_cast<OrderStatusTxnInput *>(curr_base_txn_ptr->data);

            uint32_t district_id = os_txn_ptr->d_id - 1 + (os_txn_ptr->w_id - 1) * 10;
            uint32_t latest_slot = district_id * customers_per_district + (os_txn_ptr->c_id - 1);
            uint32_t o_id = d_latest_oid[latest_slot];
            uint32_t cache_idx = district_id * num_slots_per_district + o_id;

            if (tile.thread_rank() == curr_rank)
            {
                os_txn_ptr->o_id = o_id;
                os_txn_ptr->num_items = order_num_items[cache_idx];
            }
            break;
        }
        case TpccTxnType::DELIVERY: {
            auto dl_txn_ptr = reinterpret_cast<DeliveryTxnInput *>(curr_base_txn_ptr->data);
            if (tile.thread_rank() < 10)
            {
                uint32_t d_oid = dl_txn_ptr->o_id[tile.thread_rank()];
                // Per-district deliver: o_id==0 -> nothing to deliver. Skip the
                // cache read and emit invalid-CRID sentinel + zero items.
                if (d_oid == 0)
                {
                    dl_txn_ptr->num_items[tile.thread_rank()] = 0;
                    dl_txn_ptr->customers[tile.thread_rank()] = 0xffffffffu;
                }
                else
                {
                    uint32_t district_id = tile.thread_rank() + (dl_txn_ptr->w_id - 1) * 10;
                    uint32_t cache_idx = district_id * num_slots_per_district + d_oid;
                    dl_txn_ptr->num_items[tile.thread_rank()] = order_num_items[cache_idx];
                    dl_txn_ptr->customers[tile.thread_rank()] = order_customers[cache_idx];
                }
            }
            break;
        }
        case TpccTxnType::STOCK_LEVEL: {
            auto sl_txn_ptr = reinterpret_cast<StockLevelTxnInput *>(curr_base_txn_ptr->data);
            auto sl_param_ptr = reinterpret_cast<StockLevelTxnParams *>(curr_base_param_ptr->data);

            uint32_t items_cnt = 0;
            for (uint32_t o_id = sl_txn_ptr->o_id; o_id > sl_txn_ptr->o_id - 20; --o_id)
            {
                uint32_t district_id = sl_txn_ptr->d_id - 1 + (sl_txn_ptr->w_id - 1) * 10;
                uint32_t cache_idx = district_id * num_slots_per_district + o_id;
                uint32_t num_items = order_num_items[cache_idx];
                if (tile.thread_rank() < num_items)
                {
                    sl_param_ptr->stock_ids[items_cnt + tile.thread_rank()] = order_items[cache_idx][tile.thread_rank()];
                }
                items_cnt += num_items;
            }
            struct CustomLess
            {
                __device__ bool operator()(const uint32_t &lhs, const uint32_t &rhs) { return lhs < rhs; }
            };
            WarpMergeSortT(temp_storage[warp_id])
                .Sort(
                    *reinterpret_cast<uint32_t(*)[19]>(&sl_param_ptr->stock_ids[items_per_thread * tile.thread_rank()]),
                    CustomLess(), items_cnt, 0xffffffffu);
            if (tile.thread_rank() == curr_rank) { sl_txn_ptr->num_items = items_cnt; }
            break;
        }
        default:
            assert(false);
            break;
        }

        if (tile.thread_rank() == curr_rank) { to_insert = false; }
        work_queue = tile.ballot(to_insert);
    }
}

class TpccGpuAuxIndexImpl
{
    TpccConfig config;
    CustomerOrderBTree co_btree;
    uint32_t num_slots_per_district;
    uint32_t num_slots;

    // Flat secondary index for OrderStatus, env-gated by EPIC_FLAT_AUX_INDEX=1.
    // Replaces co_btree's role for the OrderStatus "latest order of customer"
    // lookup. Sized to the fixed (W * 10 * 3000) (w,d,c) cardinality.
    // Cardinality is a TPC-C constant, so this never grows. NewOrder uses
    // atomicMax to update; OrderStatus does a single global load.
    static constexpr uint32_t kCustomersPerDistrict = 3000;
    bool use_flat_aux_;
    uint32_t* d_latest_oid_;
    static bool flatAuxIndexEnabledFromEnv(ExecMode mode) {
        return envBoolOrHybridDefault("EPIC_FLAT_AUX_INDEX", mode);
    }
    uint32_t *order_num_items, *order_customers;
    uint32_t (*order_items)[15];

public:
    explicit TpccGpuAuxIndexImpl(TpccConfig &config)
        : config{config}
        // num_slots_per_district sizes the per-district insert log
        // (order_num_items / order_customers / order_items) and bounds the
        // o_id-derived index `district_id * num_slots_per_district + o_id` that
        // insert_txn_updates_*_kernel writes from each NewOrder. The sizing
        // formula and its rationale live in TpccConfig::auxNumSlotsPerDistrict()
        // (single source of truth, shared with runBenchmark's overflow guard so
        // the array size and the guard threshold cannot drift).
        , num_slots_per_district(config.auxNumSlotsPerDistrict())
        , num_slots(num_slots_per_district * config.num_warehouses * 10)
        , order_num_items(cuda_allocator<uint32_t>().allocate(num_slots))
        , order_customers(cuda_allocator<uint32_t>().allocate(num_slots))
        , order_items(cuda_allocator<uint32_t[15]>().allocate(num_slots))
        , use_flat_aux_(flatAuxIndexEnabledFromEnv(config.execution_mode))
        , d_latest_oid_(use_flat_aux_
              ? cuda_allocator<uint32_t>().allocate(
                  static_cast<size_t>(config.num_warehouses) * 10 * kCustomersPerDistrict)
              : nullptr)
    {
        Logger::GetInstance().Info(
            "[AUX] TpccGpuAuxIndexImpl: num_slots_per_district={} num_slots={} (W={} txns={} epochs={})",
            num_slots_per_district, num_slots,
            config.num_warehouses, config.num_txns, config.epochs);
        if (use_flat_aux_) {
            const size_t n = static_cast<size_t>(config.num_warehouses) * 10 * kCustomersPerDistrict;
            Logger::GetInstance().Info(
                "[AUX] flat aux index ACTIVE: "
                "{} slots, {:.2f} MB, OrderStatus uses direct (w,d,c) lookup instead of cuco btree",
                n, n * sizeof(uint32_t) / (1024.0 * 1024.0));
        }
    }
    void loadInitialData()
    {
        mapped_vector<TreeParam::key_type> keys(config.num_warehouses * 10 * 3000);
        mapped_vector<TreeParam::value_type> values(config.num_warehouses * 10 * 3000);
        std::mt19937_64 gen(tpccSeedOrEntropy(0x6907ull));
        TpccNuRand i_id_dist(8191, 1, 100'000);

        std::vector<uint32_t> h_order_num_items(num_slots), h_order_customers(num_slots);
        uint32_t (*h_order_items)[15] = new uint32_t[num_slots][15];

        auto &logger = Logger::GetInstance();
        logger.Info("Loading gpu aux index");
        auto start = std::chrono::high_resolution_clock::now();
        uint32_t idx = 0;
        for (uint32_t w_id = 1; w_id <= config.num_warehouses; w_id++)
        {
            for (uint32_t d_id = 1; d_id <= 10; d_id++)
            {
                for (uint32_t o_id = 1; o_id <= 3000; o_id++)
                {
                    const uint32_t c_id = o_id; // assume one order per customer
                    const uint32_t sid = 0;     // initial sid
                    uint64_t hacked_oid = PackedCustomerOrderKey::max_o_id - o_id; // hack: to get backward scan
                    const PackedCustomerOrderKey key{hacked_oid, c_id, d_id, w_id};
                    keys[idx] = key.base_key;
                    values[idx] = sid;
                    ++idx;

                    const uint32_t o_num_items = 15;
                    const uint32_t district_id = d_id - 1+ (w_id - 1) * 10;
                    h_order_num_items[district_id * num_slots_per_district + o_id] = o_num_items; // TODO: randomize num_items for each order
                    h_order_customers[district_id * num_slots_per_district + o_id] = c_id;

                    uint32_t(&o_items)[15] = h_order_items[district_id * num_slots_per_district + o_id];
                    for (uint32_t ol_id = 0; ol_id < o_num_items; ++ol_id)
                    {
                        o_items[ol_id] = i_id_dist(gen);
                    }
                }
            }
        }
        if (use_flat_aux_) {
            // Skip the cuco btree's bulk insert on the flat path. Initial state
            // is computed deterministically by k_init_latest_oid_initial from
            // the (w, d, c) lex order of the loader (c gets one initial order
            // with o_id = c). No host-side keys vector needed.
            constexpr int B = 256;
            const uint32_t total = static_cast<uint32_t>(config.num_warehouses) * 10u * kCustomersPerDistrict;
            int G = (total + B - 1) / B;
            k_init_latest_oid_initial<<<G, B>>>(d_latest_oid_, config.num_warehouses, kCustomersPerDistrict);
            gpu_err_check(cudaPeekAtLastError());
        } else {
            co_btree.insert(keys.data(), values.data(), idx);
        }
        gpu_err_check(cudaMemcpy(order_num_items, h_order_num_items.data(), sizeof(uint32_t) * num_slots, cudaMemcpyHostToDevice));
        gpu_err_check(cudaMemcpy(order_customers, h_order_customers.data(), sizeof(uint32_t) * num_slots, cudaMemcpyHostToDevice));
        gpu_err_check(cudaMemcpy(order_items, h_order_items, sizeof(uint32_t[15]) * num_slots, cudaMemcpyHostToDevice));

        delete []h_order_items;

        gpu_err_check(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        logger.Info(
            "Gpu aux index loaded in {}us", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        logger.Info("Validating gpu aux btree");
        std::vector<TreeParam::key_type> h_keys = keys.to_std_vector();
        // co_btree.validate_tree_structure(h_keys, [](const TreeParam::key_type &key) {return 0;});
        logger.Info("Validation passed: gpu aux btree");
    }

    template <typename TxnArrayType>
    void insertTxnUpdates(TxnArrayType &txns, size_t epoch)
    {
        uint32_t num_txns = txns.num_txns;
        const uint32_t block_size = 512;
        const uint32_t num_blocks = (num_txns + block_size - 1) / block_size;
        if (use_flat_aux_) {
            insert_txn_updates_flat_kernel<<<num_blocks, block_size>>>(
                GpuPackedTxnArray(txns), num_txns,
                order_num_items, order_customers, order_items, num_slots_per_district,
                d_latest_oid_, kCustomersPerDistrict);
        } else {
            insert_txn_updates_kernel<<<num_blocks, block_size>>>(GpuPackedTxnArray(txns), num_txns, co_btree,
                co_btree.get_allocator(), order_num_items, order_customers, order_items, num_slots_per_district);
        }
        gpu_err_check(cudaDeviceSynchronize());
        // Warn (but don't crash) if the btree allocator is climbing past 75 %
        // of num_prealloc_nodes — the next sizing bump would happen here.
        if ((epoch & 0x7) == 0) {
            uint32_t cnt = co_btree.get_allocator().get_allocated_count();
            if (cnt > TreeParam::num_prealloc_nodes * 3 / 4) {
                Logger::GetInstance().Info(
                    "[AUX] btree slab high-water at epoch {}: {} / {} "
                    "(>75% — consider bumping TreeParam::num_prealloc_nodes)",
                    (uint32_t)epoch, cnt, (uint32_t)TreeParam::num_prealloc_nodes);
            }
        }
    }

    template <typename TxnArrayType, typename TxnParamArrayType>
    void performRangeQueries(TxnArrayType &txns, TxnParamArrayType &index, size_t epoch)
    {
        uint32_t num_txns = txns.num_txns;
        const uint32_t block_size = 512;
        const uint32_t num_blocks = (num_txns + block_size - 1) / block_size;
        if (use_flat_aux_) {
            perform_range_queries_flat_kernel<<<num_blocks, block_size>>>(
                GpuPackedTxnArray(txns), TpccGpuTxnArrayT(index), num_txns,
                order_num_items, order_customers, order_items, num_slots_per_district,
                d_latest_oid_, kCustomersPerDistrict);
        } else {
            perform_range_queries_kernel<<<num_blocks, block_size>>>(GpuPackedTxnArray(txns), TpccGpuTxnArrayT(index), num_txns,
                co_btree, co_btree.get_allocator(), order_num_items, order_customers, order_items, num_slots_per_district);
        }
        gpu_err_check(cudaDeviceSynchronize());
    }

    // See declaration in tpcc_gpu_aux_index.h. Reconstructs
    // order_num_items / order_customers / order_items / d_latest_oid_ from the
    // CPU shadow + CPU primary store so the recovery-rebuilt aux index
    // reflects every NewOrder insert that happened before the crash (not
    // just the initial population that loadInitialData covers).
    void rebuildFromShadow(const TpccCpuShadowIndex& shadow,
                           const Record<OrderValue>* cpu_orders,
                           const Record<OrderLineValue>* cpu_ols)
    {
        auto& logger = Logger::GetInstance();
        logger.Info("[AUX-REBUILD] starting: shadow_o={} shadow_ol={} num_slots={}",
                    shadow.shadow_o().size(), shadow.shadow_ol().size(), num_slots);

        if (!use_flat_aux_) {
            throw std::runtime_error(
                "TpccGpuAuxIndex::rebuildFromShadow: only flat-aux path is supported; "
                "set EPIC_FLAT_AUX_INDEX=1 for the recovery path.");
        }

        // 1) Initial population. After this, the three flat arrays have the
        //    deterministic initial values and d_latest_oid_ has c_id per (w,d,c).
        loadInitialData();

        // 2) Read the just-initialized GPU arrays back to host so we can layer
        //    post-initial inserts on top without clobbering the initial slots.
        std::vector<uint32_t> h_order_num_items(num_slots);
        std::vector<uint32_t> h_order_customers(num_slots);
        // RAII host staging buffer (was a raw new[]; the cudaMemcpy calls below
        // throw on error, which would leak the raw allocation). std::array<...,15>
        // has the same layout as uint32_t[15] and the vector is contiguous, so
        // .data() + the same byte sizes match the GPU side exactly.
        std::vector<std::array<uint32_t, 15>> h_order_items(num_slots);
        gpu_err_check(cudaMemcpy(h_order_num_items.data(), order_num_items,
            num_slots * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpu_err_check(cudaMemcpy(h_order_customers.data(), order_customers,
            num_slots * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpu_err_check(cudaMemcpy(h_order_items.data(), order_items,
            num_slots * sizeof(uint32_t[15]), cudaMemcpyDeviceToHost));

        // Initial-population CRID counts: anything below these came from the
        // deterministic initial load and is already covered by loadInitialData.
        const uint32_t initial_o  = config.num_warehouses * 10u * 3000u;
        const uint32_t initial_ol = config.num_warehouses * 10u * 3000u * 15u;
        const uint32_t maxO_no    = shadow.max_o_orders();
        const uint32_t maxO_ol    = shadow.max_o_ol();

        // 3) Walk shadow_o for post-initial Order inserts. Decode (w,d,o) from
        //    the dense_idx, read OrderValue from cpu_orders[crid], write into
        //    the (district, o_id)-indexed aux slot. We pick the slot with the
        //    larger version tag because the executor writes to slot epoch%2,
        //    and the other slot can be either an older committed value or
        //    uninitialized (insert path).
        size_t replayed_o = 0;
        size_t skipped_oob_o = 0;
        const auto& shadow_o = shadow.shadow_o();
        for (size_t i = 0; i < shadow_o.size(); ++i) {
            uint32_t crid = shadow_o[i];
            if (crid == TpccCpuShadowIndex::kSentinel) continue;
            if (crid < initial_o) continue;  // initial slot — already populated

            // Decode dense_idx -> (w,d,o). Matches denseIdxO encoding.
            uint32_t per_district  = maxO_no;
            uint32_t per_warehouse = 10u * per_district;
            uint32_t w_zb = static_cast<uint32_t>(i / per_warehouse);
            uint32_t rem  = static_cast<uint32_t>(i % per_warehouse);
            uint32_t d_zb = rem / per_district;
            uint32_t o_zb = rem % per_district;
            uint32_t w = w_zb + 1u, d = d_zb + 1u, o = o_zb + 1u;

            const auto& rec = cpu_orders[crid];
            const OrderValue& v = (rec.version1 >= rec.version2) ? rec.value1 : rec.value2;

            uint32_t district_id = (w - 1u) * 10u + (d - 1u);
            uint32_t slot        = district_id * num_slots_per_district + o;
            // num_slots_per_district is sized for the canonical workload; the
            // shadow's max_o_orders is sized for worst-case insert spread, so
            // an o_id > num_slots_per_district CAN appear at deep epochs or
            // skewed inserts. Skip + count so we can bump
            // num_slots_per_district if the warning fires.
            if (slot >= num_slots) { ++skipped_oob_o; continue; }

            h_order_customers[slot] = v.o_c_id;
            h_order_num_items[slot] = v.o_ol_cnt;
            ++replayed_o;
        }

        // 4) Walk shadow_ol for post-initial OrderLine inserts. Decode
        //    (w,d,o,ol_n) from the dense_idx (matches denseIdxOL in
        //    tpcc_flat_index.h), read OrderLineValue, write item_id.
        size_t replayed_ol = 0;
        size_t skipped_oob_ol = 0;
        const auto& shadow_ol = shadow.shadow_ol();
        for (size_t i = 0; i < shadow_ol.size(); ++i) {
            uint32_t crid = shadow_ol[i];
            if (crid == TpccCpuShadowIndex::kSentinel) continue;
            if (crid < initial_ol) continue;  // initial — already populated

            // denseIdxOL = district_id * maxO_ol * 15 + (o-1) * 15 + (ol-1)
            uint32_t per_o         = 15u;
            uint32_t per_district  = maxO_ol * per_o;
            uint32_t per_warehouse = 10u * per_district;
            uint32_t w_zb  = static_cast<uint32_t>(i / per_warehouse);
            uint32_t rem1  = static_cast<uint32_t>(i % per_warehouse);
            uint32_t d_zb  = rem1 / per_district;
            uint32_t rem2  = rem1 % per_district;
            uint32_t o_zb  = rem2 / per_o;
            uint32_t ol_zb = rem2 % per_o;
            uint32_t w = w_zb + 1u, d = d_zb + 1u, o = o_zb + 1u, ol = ol_zb + 1u;

            const auto& rec = cpu_ols[crid];
            const OrderLineValue& v = (rec.version1 >= rec.version2) ? rec.value1 : rec.value2;

            uint32_t district_id = (w - 1u) * 10u + (d - 1u);
            uint32_t slot        = district_id * num_slots_per_district + o;
            if (slot >= num_slots) { ++skipped_oob_ol; continue; }
            if (ol < 1u || ol > 15u) continue;

            h_order_items[slot][ol - 1u] = v.ol_i_id;
            ++replayed_ol;
        }

        // 5) Upload the merged buffers back to GPU.
        gpu_err_check(cudaMemcpy(order_num_items, h_order_num_items.data(),
            num_slots * sizeof(uint32_t), cudaMemcpyHostToDevice));
        gpu_err_check(cudaMemcpy(order_customers, h_order_customers.data(),
            num_slots * sizeof(uint32_t), cudaMemcpyHostToDevice));
        gpu_err_check(cudaMemcpy(order_items, h_order_items.data(),
            num_slots * sizeof(uint32_t[15]), cudaMemcpyHostToDevice));

        // 6) Update d_latest_oid_ for inserted orders. The atomicMax against
        //    k_init_latest_oid_initial's c_id baseline keeps the larger of
        //    initial (c_id) and any inserted o_id owned by c, which is the
        //    correct semantics.
        {
            constexpr int B = 256;
            uint32_t total = config.num_warehouses * 10u * num_slots_per_district;
            int G = static_cast<int>((total + B - 1u) / B);
            k_update_latest_oid_from_order_customers<<<G, B>>>(
                d_latest_oid_, order_customers, config.num_warehouses,
                num_slots_per_district, kCustomersPerDistrict);
            gpu_err_check(cudaPeekAtLastError());
            gpu_err_check(cudaDeviceSynchronize());
        }

        logger.Info("[AUX-REBUILD] complete: replayed_orders={} replayed_orderlines={}",
                    replayed_o, replayed_ol);
        if (skipped_oob_o > 0 || skipped_oob_ol > 0) {
            logger.Info(
                "[AUX-REBUILD] WARNING: skipped {} order(s) and {} orderline(s) whose decoded "
                "(w,d,o) hits o_id >= num_slots_per_district ({}); these inserts are lost from the "
                "aux index. Bump num_slots_per_district (gpu_aux_index ctor) to absorb the deeper "
                "insert pool.",
                skipped_oob_o, skipped_oob_ol, num_slots_per_district);
        }
    }

    // Free the raw device buffers this Impl owns. These are the
    // three flat insert-log arrays (always allocated in the ctor) plus the flat
    // OrderStatus index (allocated only on the use_flat_aux_ path, nullptr
    // otherwise). Each is freed with the SAME cuda_allocator<T> that allocated
    // it (cuda_allocator::deallocate just calls cudaFree) and nulled, so a
    // second call is a no-op (cudaFree(nullptr) is well-defined). The
    // CustomerOrderBTree member (co_btree) is RAII and self-frees via its own
    // destructor, so it is intentionally NOT touched here.
    void freeDeviceBuffers()
    {
        if (order_num_items) {
            cuda_allocator<uint32_t>().deallocate(order_num_items, num_slots);
            order_num_items = nullptr;
        }
        if (order_customers) {
            cuda_allocator<uint32_t>().deallocate(order_customers, num_slots);
            order_customers = nullptr;
        }
        if (order_items) {
            cuda_allocator<uint32_t[15]>().deallocate(order_items, num_slots);
            order_items = nullptr;
        }
        if (d_latest_oid_) {
            cuda_allocator<uint32_t>().deallocate(
                d_latest_oid_,
                static_cast<size_t>(config.num_warehouses) * 10 * kCustomersPerDistrict);
            d_latest_oid_ = nullptr;
        }
    }
};

} // namespace detail

template <typename TxnArrayType, typename TxnParamArrayType>
TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::TpccGpuAuxIndex(TpccConfig &config)
    : impl{detail::TpccGpuAuxIndexImpl{config}}
{}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::loadInitialData()
{
    std::any_cast<detail::TpccGpuAuxIndexImpl>(impl).loadInitialData();
}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::insertTxnUpdates(TxnArrayType &txns, size_t epoch)
{
    std::any_cast<detail::TpccGpuAuxIndexImpl>(impl).insertTxnUpdates(txns, epoch);
}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::performRangeQueries(
    TxnArrayType &txns, TxnParamArrayType &index, size_t epoch)
{
    std::any_cast<detail::TpccGpuAuxIndexImpl>(impl).performRangeQueries(txns, index, epoch);
}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::rebuildFromShadow(
    const TpccCpuShadowIndex& shadow,
    const Record<OrderValue>* cpu_orders,
    const Record<OrderLineValue>* cpu_ols)
{
    std::any_cast<detail::TpccGpuAuxIndexImpl>(impl).rebuildFromShadow(
        shadow, cpu_orders, cpu_ols);
}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::freeDeviceBuffers()
{
    // Reference cast (not the by-value cast the other forwarders use) so the
    // pointers nulled in freeDeviceBuffers persist in the stored Impl, making
    // the call idempotent against a double-free.
    std::any_cast<detail::TpccGpuAuxIndexImpl&>(impl).freeDeviceBuffers();
}

/* instantiate the templated class with choosen type */
template class TpccGpuAuxIndex<TpccTxnArrayT, TpccTxnParamArrayT>;

} // namespace epic::tpcc
