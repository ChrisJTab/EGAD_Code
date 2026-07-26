// crid_grid_index.cu
// Implementation for CRID -> (GRID, last_epoch, dirty) index

// USE_FLAT_ARRAY_INDEX is defined in crid_grid_index.h

#include <benchmarks/crid_grid_index.h>

#include <gpu_allocator.h>
#include <util_gpu_error_check.cuh>

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <gpu_txn.cuh>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <cub/device/device_select.cuh>

namespace epic::ycsb {

struct CridGridIndex::Impl {
    uint32_t* d_crid_to_grid   = nullptr;  // flat array [num_units], value = GRID or the 0xffffffff empty sentinel
    uint32_t  num_units        = 0;
    // TODO (cleanup): grid_epoch_last is written by k_touch_by_grids but
    // no consumer reads it (cursor-based FIFO eviction does not need it).
    // Remove with k_touch_by_grids, CridGridIndex::touch_by_grids, and the
    // call site in HybridStager::prepareEpoch if the LRU policy is not coming back.
    uint32_t* grid_epoch_last  = nullptr;
    uint8_t* grid_dirty_v1     = nullptr;
    uint8_t* grid_dirty_v2     = nullptr;
    uint32_t  capacity         = 0;

    // collect_dirty_grids cached scratch
    void*     d_select_tmp      = nullptr;
    size_t    select_tmp_bytes  = 0;
    uint32_t* d_num_selected     = nullptr;   // device scalar
    Allocator* alloc            = nullptr;    // store allocator ptr for freeing
};

// ============================================================
// Flat-array kernels (USE_FLAT_ARRAY_INDEX path)
// ============================================================

namespace {

__global__ void k_flat_insert(const uint32_t* crids, const uint32_t* grids, uint32_t n,
                              uint32_t* d_crid_to_grid, uint32_t num_units)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t crid = crids[i];
    // The caller's d_new_ / d_evict_crids buffers can carry sentinel CRIDs
    // that propagated through the per-epoch dedup, and the CRID space is
    // bounded by num_units. Guarding both here keeps the d_crid_to_grid
    // write in-bounds. Without the guard, sentinel CRIDs write 4 GiB past
    // the array (or past num_units for stale CRIDs from a smaller table),
    // corrupting unrelated GPU heap state and surfacing as illegal-access
    // in whichever kernel uses that memory next.
    if (crid != 0xffffffffu && crid < num_units) {
        d_crid_to_grid[crid] = grids[i];
    }
}

__global__ void k_flat_lookup_and_mark_needed(const uint32_t* crids, uint32_t* out_grids, uint32_t n,
                                              const uint32_t* d_crid_to_grid, uint32_t num_units,
                                              uint32_t cap, uint8_t* needed_flag)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t crid = crids[i];
    uint32_t g = (crid < num_units) ? d_crid_to_grid[crid] : 0xffffffffu;
    out_grids[i] = g;
    if (g != 0xffffffffu && g < cap) {
        needed_flag[g] = 1;
    }
}


__global__ void k_flat_rename_crids(const uint32_t* old_crids, const uint32_t* new_crids,
                                    const uint32_t* grids, uint32_t n,
                                    uint32_t* d_crid_to_grid, uint32_t num_units,
                                    uint8_t* grid_dirty_v1, uint8_t* grid_dirty_v2)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t oldc = old_crids[i];
    uint32_t newc = new_crids[i];
    uint32_t g    = grids[i];
    // Same bounds reasoning as k_flat_insert above; both old and new CRIDs
    // can be sentinel-padded slots from incomplete txns.
    if (oldc != 0xffffffffu && oldc < num_units) {
        d_crid_to_grid[oldc] = 0xffffffffu;
    }
    if (newc != 0xffffffffu && newc < num_units) {
        d_crid_to_grid[newc] = g;
    }
}

__global__ void k_flat_remap_txn_record_ids(GpuTxnArray txns, uint32_t num_txns,
                                            const uint32_t* d_crid_to_grid, uint32_t num_units,
                                            uint32_t staging_capacity,
                                            bool skip_multi_field_ops)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns) return;

    BaseTxn* base = txns.getTxn(tid);
    auto* p = reinterpret_cast<ycsb::YcsbTxnParam*>(base->data);

#pragma unroll
    for (int i = 0; i < 10; ++i) {
        auto op = p->ops[i];
        // In split-field mode, INSERT / FULL_READ / FULL_READ_MODIFY_WRITE need all 10
        // per-field GRIDs, not one, so remap is skipped for them and record_ids[i]
        // stays as logical R for fill_{insert,full_read}_field_grids to consume.
        // In non-split mode there is only one slot per record, so every op (including
        // INSERT / FULL_READ / FULL_RMW) reads record_ids[i] as a single GRID; remap
        // must run for all op types.
        if (skip_multi_field_ops) {
            if (op == ycsb::YcsbOpType::INSERT) continue;
            if (op == ycsb::YcsbOpType::FULL_READ) continue;
            if (op == ycsb::YcsbOpType::FULL_READ_MODIFY_WRITE) continue;
        }

        uint32_t crid = p->record_ids[i];
        uint32_t g = (crid < num_units) ? d_crid_to_grid[crid] : 0xffffffffu;
        p->record_ids[i] = g;
    }
}

// For each FULL_READ / FULL_READ_MODIFY_WRITE op, expand the logical record id R
// (still stored in record_ids[i] — it was skipped by promote and remap) into 10
// per-field GRIDs via d_crid_to_grid[R*10 + j] and write them into the plan.
__global__ void k_flat_fill_full_read_field_grids(GpuTxnArray txns, GpuTxnArray plans,
                                                  const uint32_t* d_crid_to_grid, uint32_t num_units,
                                                  uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns) return;

    auto* tp = reinterpret_cast<ycsb::YcsbTxnParam*>(txns.getTxn(tid)->data);
    auto* ep = reinterpret_cast<ycsb::YcsbExecPlan*>(plans.getTxn(tid)->data);

#pragma unroll
    for (int i = 0; i < 10; ++i) {
        auto op = tp->ops[i];
        if (op != ycsb::YcsbOpType::FULL_READ &&
            op != ycsb::YcsbOpType::FULL_READ_MODIFY_WRITE) continue;

        uint32_t rec_crid = tp->record_ids[i]; // logical R (promote/remap skipped us)

#pragma unroll
        for (int f = 0; f < 10; ++f) {
            uint32_t fcrid = rec_crid * 10u + (uint32_t)f;
            uint32_t grid = (fcrid < num_units) ? d_crid_to_grid[fcrid] : 0xffffffffu;
            if (op == ycsb::YcsbOpType::FULL_READ) {
                ep->plans[i].full_read_plan.field_grids[f] = grid;
            } else {
                ep->plans[i].full_read_modify_write_plan.field_grids[f] = grid;
            }
        }
    }
}

__global__ void k_flat_fill_insert_field_grids(GpuTxnArray txns, GpuTxnArray plans,
                                               const uint32_t* d_crid_to_grid, uint32_t num_units,
                                               uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns) return;

    auto* tp = reinterpret_cast<ycsb::YcsbTxnParam*>(txns.getTxn(tid)->data);
    auto* ep = reinterpret_cast<ycsb::YcsbExecPlan*>(plans.getTxn(tid)->data);

#pragma unroll
    for (int i = 0; i < 10; ++i) {
        if (tp->ops[i] != ycsb::YcsbOpType::INSERT) continue;

        uint32_t rec_crid = tp->record_ids[i];

#pragma unroll
        for (int f = 0; f < 10; ++f) {
            uint32_t fcrid = rec_crid * 10u + (uint32_t)f;
            uint32_t grid = (fcrid < num_units) ? d_crid_to_grid[fcrid] : 0xffffffffu;
            ep->plans[i].insert_full_plan.field_grids[f] = grid;
        }
    }
}

} // anon namespace (flat-array kernels)


// ---- Kernels shared by both paths ----
namespace {

// TODO (cleanup): k_touch_by_grids writes grid_epoch_last but no consumer
// reads it. The cursor-based FIFO eviction does not need it. Remove this
// kernel, the touch_by_grids wrapper below, the grid_epoch_last allocation
// in CridGridIndex::Impl, and the call site in HybridStager::prepareEpoch
// if the LRU eviction policy is not coming back.
__global__ void k_touch_by_grids(const uint32_t* __restrict__ grids,
                             uint32_t n,
                             uint32_t epoch,
                             uint32_t cap,
                             uint32_t* __restrict__ grid_epoch_last)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t g = grids[i];
    if (g == 0xffffffffu || g >= cap) return;   // skip misses
    grid_epoch_last[g] = epoch;
}

} // anon namespace


// ====================== CridGridIndex impl ======================

CridGridIndex::CridGridIndex(Allocator& alloc, uint32_t capacity, uint32_t num_units, double load_factor)
    : impl_(std::make_unique<Impl>())
{
    impl_->alloc = &alloc;
    impl_->capacity = capacity;
    impl_->num_units = num_units;

    // 1) allocate flat CRID->GRID array and fill with empty sentinel
    impl_->d_crid_to_grid = static_cast<uint32_t*>(alloc.Allocate(sizeof(uint32_t) * num_units));
    gpu_err_check(cudaMemset(impl_->d_crid_to_grid, 0xff, sizeof(uint32_t) * num_units));

    // 2) allocate metadata arrays (per GRID slot)
    impl_->grid_epoch_last = static_cast<uint32_t*>(alloc.Allocate(sizeof(uint32_t) * impl_->capacity));
    impl_->grid_dirty_v1      = static_cast<uint8_t* >(alloc.Allocate(sizeof(uint8_t ) * impl_->capacity));
    impl_->grid_dirty_v2      = static_cast<uint8_t* >(alloc.Allocate(sizeof(uint8_t ) * impl_->capacity));

    // 3) initialize metadata (epoch=0, dirty=0)
    gpu_err_check(cudaMemset(impl_->grid_epoch_last, 0, sizeof(uint32_t) * impl_->capacity));
    gpu_err_check(cudaMemset(impl_->grid_dirty_v1,      0, sizeof(uint8_t ) * impl_->capacity));
    gpu_err_check(cudaMemset(impl_->grid_dirty_v2,      0, sizeof(uint8_t ) * impl_->capacity));

    // 4) allocate scratch for collect_dirty_grids
    impl_->d_num_selected = static_cast<uint32_t*>(alloc.Allocate(sizeof(uint32_t)));
    gpu_err_check(cudaMemset(impl_->d_num_selected, 0, sizeof(uint32_t)));
}

CridGridIndex::~CridGridIndex()
{
    if (!impl_) return;

    if (impl_->d_select_tmp) cudaFree(impl_->d_select_tmp);

    if (impl_->alloc) {
        if (impl_->d_crid_to_grid) impl_->alloc->Free(impl_->d_crid_to_grid);
        if (impl_->d_num_selected)  impl_->alloc->Free(impl_->d_num_selected);
        if (impl_->grid_epoch_last) impl_->alloc->Free(impl_->grid_epoch_last);
        if (impl_->grid_dirty_v1)      impl_->alloc->Free(impl_->grid_dirty_v1);
        if (impl_->grid_dirty_v2)      impl_->alloc->Free(impl_->grid_dirty_v2);
    }
}


CridGridIndexView CridGridIndex::device_view() const {
    CridGridIndexView v{};
    v.grid_epoch_last = impl_->grid_epoch_last;
    v.grid_dirty_v1      = impl_->grid_dirty_v1;
    v.grid_dirty_v2      = impl_->grid_dirty_v2;
    v.capacity        = impl_->capacity;
    v.bucket_count    = 0;
    v.d_crid_to_grid  = impl_->d_crid_to_grid;
    v.num_units       = impl_->num_units;
    return v;
}

__global__ void k_clear_dirty(const uint32_t* __restrict__ grids,
                              const uint8_t* __restrict__ slots,
                              uint32_t n,
                              uint8_t* __restrict__ grid_dirty_v1,
                              uint8_t* __restrict__ grid_dirty_v2) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint32_t g = grids[i];

    uint8_t s = slots[i];
    if (s == 0) {
        grid_dirty_v1[g] = 0;
    } else {
        grid_dirty_v2[g] = 0;
    }
}

namespace {
struct is_dirty {
    const uint8_t* dirty;
    __device__ bool operator()(const uint32_t& g) const { return dirty[g] != 0; }
};
}

uint32_t CridGridIndex::collect_dirty_grids_v1(uint32_t* d_out_grids /*size >= capacity*/, cudaStream_t stream) const {
    // Build [0..capacity) on device:
    const uint32_t cap = impl_->capacity;
    if (cap == 0) return 0;

    auto in_begin = thrust::make_counting_iterator<uint32_t>(0);
    if (impl_->select_tmp_bytes == 0) {
        // First-time allocation of scratch
        cub::DeviceSelect::Flagged(
            /*d_temp_storage*/ nullptr,
            /*temp_storage_bytes*/ impl_->select_tmp_bytes,
            /*d_in*/ in_begin,
            /*d_flags*/ impl_->grid_dirty_v1,   // uint8_t flags (0/1) works
            /*d_out*/ d_out_grids,
            /*d_num_selected_out*/ impl_->d_num_selected,
            /*num_items*/ cap,
            /*stream*/ stream
        );
        gpu_err_check(cudaMalloc(&impl_->d_select_tmp, impl_->select_tmp_bytes));
    }

    cub::DeviceSelect::Flagged(
        impl_->d_select_tmp,
        impl_->select_tmp_bytes,
        in_begin,
        impl_->grid_dirty_v1,
        d_out_grids,
        impl_->d_num_selected,
        cap,
        stream
    );
    uint32_t h = 0;
    gpu_err_check(cudaMemcpyAsync(&h, impl_->d_num_selected, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    gpu_err_check(cudaStreamSynchronize(stream));
    return h;
}

    uint32_t CridGridIndex::collect_dirty_grids_v2(uint32_t* d_out_grids /*size >= capacity*/, cudaStream_t stream) const {
    // Build [0..capacity) on device:
    const uint32_t cap = impl_->capacity;
    if (cap == 0) return 0;

    auto in_begin = thrust::make_counting_iterator<uint32_t>(0);
    if (impl_->select_tmp_bytes == 0) {
        // First-time allocation of scratch
        cub::DeviceSelect::Flagged(
            /*d_temp_storage*/ nullptr,
            /*temp_storage_bytes*/ impl_->select_tmp_bytes,
            /*d_in*/ in_begin,
            /*d_flags*/ impl_->grid_dirty_v2,   // uint8_t flags (0/1) works
            /*d_out*/ d_out_grids,
            /*d_num_selected_out*/ impl_->d_num_selected,
            /*num_items*/ cap,
            /*stream*/ stream
        );
        gpu_err_check(cudaMalloc(&impl_->d_select_tmp, impl_->select_tmp_bytes));
    }

    cub::DeviceSelect::Flagged(
        impl_->d_select_tmp,
        impl_->select_tmp_bytes,
        in_begin,
        impl_->grid_dirty_v2,
        d_out_grids,
        impl_->d_num_selected,
        cap,
        stream
    );
    uint32_t h = 0;
    gpu_err_check(cudaMemcpyAsync(&h, impl_->d_num_selected, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    gpu_err_check(cudaStreamSynchronize(stream));
    return h;
}

void CridGridIndex::bulk_insert(const uint32_t* d_crids,
                                const uint32_t* d_grids,
                                uint32_t n,
                                cudaStream_t stream)
{
    if (n == 0 || d_crids == nullptr || d_grids == nullptr) return;

    constexpr int B = 256;
    int G = (n + B - 1) / B;
    k_flat_insert<<<G, B, 0, stream>>>(d_crids, d_grids, n, impl_->d_crid_to_grid, impl_->num_units);
    gpu_err_check(cudaPeekAtLastError());
}

void CridGridIndex::clear_dirty_by_grids(const uint32_t* d_grids, const uint8_t* d_slots, uint32_t n,
                                         cudaStream_t stream) {
    if (n==0) return;
    constexpr int B=256; int G=(n+B-1)/B;
    k_clear_dirty<<<G,B, 0, stream>>>(d_grids, d_slots, n, impl_->grid_dirty_v1, impl_->grid_dirty_v2);
    gpu_err_check(cudaPeekAtLastError());
}



// Replace old_crids[i] -> grids[i] with new_crids[i] -> grids[i], in-place.
void CridGridIndex::bulk_update_crid(const uint32_t* d_old_crids,
                                     const uint32_t* d_new_crids,
                                     const uint32_t* d_grids,
                                     uint32_t n,
                                     cudaStream_t stream)
{
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);

    k_flat_rename_crids<<<grid, block, 0, stream>>>(
        d_old_crids, d_new_crids, d_grids, n,
        impl_->d_crid_to_grid, impl_->num_units,
        impl_->grid_dirty_v1, impl_->grid_dirty_v2);
    gpu_err_check(cudaPeekAtLastError());
    // No host-side insert needed; everything happened in the kernel.
}

void CridGridIndex::remap_txn_record_ids(epic::TxnArray<ycsb::YcsbTxnParam>& txns,
                                         uint32_t num_txns,
                                         uint32_t staging_capacity,
                                         bool split_field)
{
    if (num_txns == 0) return;

    // Wrap the TxnArray with the device accessor the executor already uses.
    GpuTxnArray d_txns(txns); // zero-copy wrapper over device memory

    constexpr int block = 256;
    int grid = (int)((num_txns + block - 1) / block);

    // In split-field mode, skip INSERT/FULL_READ/FULL_RMW (they need per-field GRIDs
    // via fill_{insert,full_read}_field_grids). In non-split mode, remap every op.
    k_flat_remap_txn_record_ids<<<grid, block>>>(d_txns, num_txns,
        impl_->d_crid_to_grid, impl_->num_units, staging_capacity,
        /*skip_multi_field_ops=*/split_field);
    gpu_err_check(cudaPeekAtLastError());
}

void CridGridIndex::fill_insert_field_grids(epic::TxnArray<ycsb::YcsbTxnParam>& txns,
                                            epic::TxnArray<ycsb::YcsbExecPlan>& plans,
                                            uint32_t num_txns)
{
    if (num_txns == 0) return;

    GpuTxnArray d_txns(txns);
    GpuTxnArray d_plans(plans);

    constexpr int B = 256;
    int G = (int)((num_txns + B - 1) / B);

    k_flat_fill_insert_field_grids<<<G, B>>>(d_txns, d_plans, impl_->d_crid_to_grid, impl_->num_units, num_txns);
    gpu_err_check(cudaPeekAtLastError());
    gpu_err_check(cudaDeviceSynchronize());
}

void CridGridIndex::fill_full_read_field_grids(epic::TxnArray<ycsb::YcsbTxnParam>& txns,
                                               epic::TxnArray<ycsb::YcsbExecPlan>& plans,
                                               uint32_t num_txns)
{
    if (num_txns == 0) return;

    GpuTxnArray d_txns(txns);
    GpuTxnArray d_plans(plans);

    constexpr int B = 256;
    int G = (int)((num_txns + B - 1) / B);

    k_flat_fill_full_read_field_grids<<<G, B>>>(d_txns, d_plans, impl_->d_crid_to_grid, impl_->num_units, num_txns);
    gpu_err_check(cudaPeekAtLastError());
    gpu_err_check(cudaDeviceSynchronize());
}

// TODO (cleanup): see TODO on k_touch_by_grids above. This wrapper writes
// grid_epoch_last but nothing reads it; remove with the kernel and the
// grid_epoch_last storage if the LRU policy is not coming back.
void CridGridIndex::touch_by_grids(const uint32_t* d_grids, uint32_t n, uint32_t epoch,
                                   cudaStream_t stream)
{
    if (n == 0) return;
    constexpr int B = 256;
    int G = (n + B - 1) / B;
    k_touch_by_grids<<<G,B, 0, stream>>>(d_grids, n, epoch, impl_->capacity, impl_->grid_epoch_last);
    gpu_err_check(cudaPeekAtLastError());
}

void CridGridIndex::bulk_lookup_crid_to_grid_and_mark_needed(const uint32_t* d_crids,
                                                            uint32_t* d_out_grids,
                                                            uint32_t n,
                                                            uint8_t* d_needed_flag,
                                                            uint32_t cap,
                                                            cudaStream_t stream)
{
    if (n == 0) return;

    constexpr int block = 256;
    int grid = (int)((n + block - 1) / block);

    k_flat_lookup_and_mark_needed<<<grid, block, 0, stream>>>(
        d_crids, d_out_grids, n, impl_->d_crid_to_grid, impl_->num_units, cap, d_needed_flag);

    gpu_err_check(cudaPeekAtLastError());
}



} // namespace epic::ycsb
