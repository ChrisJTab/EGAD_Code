//
// Created by Shujian Qian on 2023-10-08.
//

#ifndef UTIL_GPU_TRANSFER_H
#define UTIL_GPU_TRANSFER_H

#include <any>

#ifdef EPIC_CUDA_AVAILABLE

namespace epic {

std::any createGpuStream();

void destroyGpuStream(std::any &stream);

void transferCpuToGpu(void *dst, const void *src, size_t size, std::any &stream);
void transferCpuToGpu(void *dst, const void *src, size_t size);

void transferGpuToCpu(void *dst, const void *src, size_t size, std::any &stream);
void transferGpuToCpu(void *dst, const void *src, size_t size);

void syncGpuStream(std::any &stream);

// Set the CUDA host-side sync mode to Yield. Makes cudaStreamSynchronize /
// cudaEventSynchronize busy-loop with sched_yield() instead of parking the
// thread on a kernel-driver interrupt. Sidesteps the ~1 ms wake-up latency
// that IRQ-based parking incurs when the calling CPU core has dropped into
// deep C-state, which dominates sync wall-clock for sub-millisecond GPU
// work (e.g. idx_transfer H2D at 120 B split and flush_sync_prev at 120 B
// non-split, where it added ~2.5-3 ms per epoch under the default Auto mode).
// Safe to call after context creation (CUDA accepts the flag override).
void setDeviceScheduleYield();

} // namespace epic

#endif // EPIC_CUDA_AVAILABLE

#endif // UTIL_GPU_TRANSFER_H
