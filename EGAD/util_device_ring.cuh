//
// Created by Christian Tabbah on 2025-09-22.
//
// This file is used to implement a ring buffer on the GPU device memory.
//Safe if producers/consumers are in different kernels or you serialize pushes.


#ifndef EPIC__UTIL_DEVICE_RING_CUH
#define EPIC__UTIL_DEVICE_RING_CUH

#include <stdint.h>
#include "device_ring_types.h"

namespace epic {
// Device-side ring for uint32_t IDs(GRIDs or CRIDs)
// struct DeviceRing
// {
//     uint32_t* buffer;      // Pointer to the ring buffer in device memory
//     unsigned long long* head; // Pointer to the head index in device memory(How many items have been popped so far, points to next pop)
//     unsigned long long* tail; // Pointer to the tail index in device memory(How many items have been pushed so far, points to next push)
//     // Count of items in the queue is tail - head
//     // we map a virtual index to physical slot by index % capacity
//     size_t capacity;       // Maximum number of elements in the ring
// };


/*
 *Example usage:
 * head = 0, tail = 0, capacity = 8
 * You push 3 items:
 * head = 0, tail = 3, capacity = 8
 * You pop 2 items:
 * head = 2, tail = 3, capacity = 8
 * You push 4 items:
 * head = 2, tail = 7, capacity = 8
 * You pop 5 items:
 * head = 7, tail = 7, capacity = 8 (tail - head = 0, empty)
 * You push 6 items:
 * head = 7, tail = 13, capacity = 8
 * The physical layout of the buffer is:
 * [item2, item3, item4, item5, item6, empty, empty, item1]
 * The logical layout of the buffer is:
 * [item1, item2, item3, item4, item5, item6, empty, empty]
 * The mapping from logical index to physical index is: physical_index = logical_index % capacity
 * The ring is empty when tail == head
 * The ring is full when tail - head == capacity
 */

#ifdef EPIC_CUDA_AVAILABLE
// Phase 1 of launch_ring_pop_parallel: single thread does the atomic head
// reservation, publishes the grant count to *granted and the old head to
// ring.pop_scratch_start so Phase 2 can read them back. Kept to 1 block / 1
// thread because the reservation is inherently serial (atomicAdd on head).
static __global__ void k_ring_pop_reserve(DeviceRingView ring,
                                          uint32_t n,
                                          uint32_t* __restrict__ granted) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned long long old_head = atomicAdd(ring.head, (unsigned long long)n);
    unsigned long long tail_now = atomicAdd(ring.tail, 0ULL);
    unsigned long long avail    = tail_now - old_head;
    uint32_t g = (avail < (unsigned long long)n) ? (uint32_t)avail : n;
    if (g < n) atomicAdd(ring.head, (unsigned long long)(-(long long)(n - g)));
    *ring.pop_scratch_start = old_head;
    *granted                = g;
}

// Phase 2: N blocks copy ring.buffer[(start + i) % cap] -> out[i] for
// i in [0, *granted). Each warp of 32 threads touches 32 contiguous
// positions in both ring.buffer and out, so reads and writes are coalesced.
// Multi-block so the work spans many SMs (the previous single-block
// version saturated one SM's memory channel at ~40 GB/s).
static __global__ void k_ring_pop_copy(DeviceRingView ring,
                                       const uint32_t* __restrict__ granted,
                                       uint32_t* __restrict__ out) {
    uint32_t g = *granted;
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g) return;
    unsigned long long start = *ring.pop_scratch_start;
    uint32_t cap = ring.capacity;
    uint32_t physical_idx = (uint32_t)((start + i) % cap);
    out[i] = ring.buffer[physical_idx];
}

static inline void launch_ring_pop_parallel(DeviceRingView ring,
                              uint32_t n,
                              uint32_t* d_out,
                              uint32_t* d_granted,
                              cudaStream_t stream) {
    if (n == 0) {
        cudaMemsetAsync(d_granted, 0, sizeof(uint32_t), stream);
        return;
    }
    k_ring_pop_reserve<<<1, 1, 0, stream>>>(ring, n, d_granted);
    constexpr int block = 256;
    int grid = (int)((n + block - 1) / block);
    k_ring_pop_copy<<<grid, block, 0, stream>>>(ring, d_granted, d_out);
}

#endif // EPIC_CUDA_AVAILABLE
} // namespace epic
#endif // EPIC__UTIL_DEVICE_RING_CUH