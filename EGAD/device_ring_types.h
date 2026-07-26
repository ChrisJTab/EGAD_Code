#pragma once
#include <cstdint>

namespace epic {
struct DeviceRingView {
    uint32_t*            buffer;
    unsigned long long*  head;
    unsigned long long*  tail;
    uint32_t             capacity;
    // 8 bytes of device scratch used by the 2-phase parallel ring_pop in
    // util_device_ring.cuh. Caller allocates once and assigns into the view.
    // nullptr is fine for any consumer that does not invoke launch_ring_pop_parallel.
    unsigned long long*  pop_scratch_start = nullptr;
};
} // namespace epic