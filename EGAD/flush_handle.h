#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace epic::ycsb {

    // Opaque CUDA event handle, kept void* so this header pulls no CUDA includes.
    using EventHandle = void*;

    struct FlushHandle {
        bool valid = false;

        // Flush-set inputs the worker scatter reads (the matching GRIDs live
        // in d_grids below; the set also serves as the eviction pin source).
        std::vector<uint32_t> h_crids;
        std::vector<uint8_t>  h_slots;   // per-entry version-slot index (0 = slot 1, 1 = slot 2)

        //used to clear dirty after the writeback
        uint32_t* d_grids = nullptr;
        uint8_t*  d_slots = nullptr;
        uint32_t  n = 0;
        uint32_t  d_cap = 0;  // allocated capacity for d_grids/d_slots (reuse across epochs)

        // Scatter worker thread (pinned to node 1, does CPU scatter)
        std::thread worker;
    };

} // namespace epic::ycsb
