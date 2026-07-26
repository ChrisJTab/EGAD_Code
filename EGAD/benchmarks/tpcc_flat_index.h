// tpcc_flat_index.h
//
// Flat-array replacement for the OrderLine cuco hash map index.
// Replaces a cuco::static_map<OrderLineKey::baseType, uint32_t> (which
// uses ~32 B per logical entry: 16 B per slot at load_factor=0.5) with a
// uint32_t[full_table_size] indexed by a dense linear function of the
// (w, d, o, ol) key fields. 8x smaller, no hashing, single global memory
// access per lookup.
//
// Selected by flatOLEnabled() in tpcc_gpu_index.cu's constructor: on by
// default for hybrid_staging, off otherwise, with EPIC_FLAT_INDEX_OL as an
// explicit override. When off, the cuco code path runs.

#ifndef TPCC_FLAT_INDEX_H
#define TPCC_FLAT_INDEX_H

#include "benchmarks/tpcc_config.h"
#include "benchmarks/tpcc_table.h"
#include <cstdint>
#include <cstdlib>

// Host/device function annotations: defined by nvcc when compiling a .cu
// translation unit. When this header is pulled in from a plain C++ file
// (e.g. tpcc_cpu_shadow_index.cpp), they must be empty so g++ does not
// see unknown identifiers. The guards leave nvcc's definitions intact.
#ifndef __host__
#  define __host__
#endif
#ifndef __device__
#  define __device__
#endif
#ifndef __forceinline__
#  define __forceinline__ inline
#endif

namespace epic::tpcc {

// True when the GPU flat-OL index path is selected. Defaults on for
// hybrid_staging and off otherwise; EPIC_FLAT_INDEX_OL overrides (=1/=0).
// Reads the env var on every call (fast and inputs do not change at run time).
inline bool flatOLEnabled(ExecMode mode)
{
    return envBoolOrHybridDefault("EPIC_FLAT_INDEX_OL", mode);
}

// Per-(w,d) OrderLine dense-index stride for the flat-OL path. Derived
// from cfg.orderLineTableSize(): max_o = ceil(table_size / (W * 10 * 15)).
// Picks up the mix-realistic pool sizing (EPIC_MIX_REALISTIC_SIZING)
// when set, rather than the worst-case formula. Shared by the GPU flat
// index allocation and the CPU shadow OL encoding so both use the same
// dense_idx -> slot mapping.
inline uint32_t computeOLMaxO(const TpccConfig& cfg)
{
    const uint64_t denom = static_cast<uint64_t>(cfg.num_warehouses) * 10ull * 15ull;
    const uint64_t total = static_cast<uint64_t>(cfg.orderLineTableSize());
    return static_cast<uint32_t>((total + denom - 1ull) / denom);
}

// Compute the dense linear index for an OrderLine row. Caller supplies
// max_o (== 3000 + run-time insert headroom per district), which sets
// the per-(w,d) stride. Initial-population rows (o in [1, 3000]) and
// run-time-inserted rows (o > 3000) share the same array; the array is
// sized to W * 10 * max_o * 15 entries.
__host__ __device__ __forceinline__ uint32_t denseIdxOL(
    uint32_t w, uint32_t d, uint32_t o, uint32_t ol, uint32_t max_o)
{
    // (w-1)*10 + (d-1) collapses (w,d) into a single district id,
    // which then strides by max_o * 15 across the array.
    return ((w - 1u) * 10u + (d - 1u)) * max_o * 15u
         + (o - 1u) * 15u
         + (ol - 1u);
}

__host__ __device__ __forceinline__ uint32_t denseIdxOL(
    OrderLineKey key, uint32_t max_o)
{
    return denseIdxOL(key.ol_w_id, key.ol_d_id, key.ol_o_id, key.ol_number, max_o);
}

// Device-side view of the OrderLine flat index. Lives in the
// tpccGpuIndexFindView struct passed to the indexing kernel. When the
// flat path is inactive this is default-constructed (d_array=nullptr,
// max_o=0) and never read.
struct OrderLineFlatView
{
    uint32_t* d_array = nullptr;  // size = W * 10 * max_o * 15
    uint32_t  max_o   = 0;

    __host__ __device__ __forceinline__
    uint32_t find(OrderLineKey key) const {
        return d_array[denseIdxOL(key, max_o)];
    }

    __host__ __device__ __forceinline__
    void insert(OrderLineKey key, uint32_t value) const {
        d_array[denseIdxOL(key, max_o)] = value;
    }
};

// Bulk insert kernel for the OL flat index. Used by InsertNewOrder
// (run-time NewOrder inserts) which already has the keys array on
// device. keys[i] may be the sentinel (uint64_t)-1 from the per-txn
// submitter (see prepareTpccIndexKernel); skip those.
#ifdef __CUDACC__
__global__ inline void k_flat_ol_bulk_insert(
    const OrderLineKey::baseType* __restrict__ keys,
    const uint32_t* __restrict__ values,
    uint32_t n,
    uint32_t* __restrict__ d_array,
    uint32_t max_o)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    OrderLineKey::baseType raw = keys[i];
    if (raw == static_cast<OrderLineKey::baseType>(-1)) return;  // sentinel
    OrderLineKey key;
    key.base_key = raw;
    d_array[denseIdxOL(key, max_o)] = values[i];
}

// Transient-free initial-population init for the OL flat index. The CPU
// loader assigns CRIDs in (w, d, o, ol) lex order via thrust::sequence,
// so the CRID for each initial row is a deterministic function of the
// thread index and we don't need a keys/values array on device at all.
// Decodes tid -> (w, d, o, ol), writes flat[denseIdxOL(...)] = tid.
//
// Saves about 1.5 GB of GPU transient at W=128 E=200 vs the older
// approach of building a uint64_t key vector on host, uploading it, and
// running k_flat_ol_bulk_insert over it. That transient was the binding
// constraint for thrust's load-time allocations.
__global__ inline void k_flat_ol_init_initial_pop(
    uint32_t* __restrict__ d_array,
    uint32_t num_warehouses,
    uint32_t max_o)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = num_warehouses * 10u * 3000u * 15u;
    if (tid >= total) return;
    // Decode tid into 0-indexed (w, d, o, ol) using the same lex order
    // as the host loop in tpcc_gpu_index.cu's loadInitialData.
    uint32_t ol = tid % 15u;
    uint32_t o  = (tid / 15u) % 3000u;
    uint32_t d  = (tid / (15u * 3000u)) % 10u;
    uint32_t w  = tid / (15u * 3000u * 10u);
    // Encode dense_idx with the configured max_o (which can be larger
    // than 3000 to reserve room for run-time inserts).
    uint32_t dense_idx = (w * 10u + d) * max_o * 15u + o * 15u + ol;
    // CRID == tid (since the loader does thrust::sequence(values, 0)).
    d_array[dense_idx] = tid;
}
#endif // __CUDACC__

} // namespace epic::tpcc

#endif // TPCC_FLAT_INDEX_H
