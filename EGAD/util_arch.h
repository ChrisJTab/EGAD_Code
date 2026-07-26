//
// Created by Shujian Qian on 2023-08-13.
//
// This file is used to define the configuration of the cache and warp size architecture for the Tpcc benchmark

#ifndef UTIL_ARCH_H
#define UTIL_ARCH_H

#include <cstdint>

namespace epic {
constexpr size_t kHostCacheLineSize = 64;
constexpr size_t kDeviceWarpSize = 32;
constexpr size_t kDeviceCacheLineSize = 128;
} // namespace epic

#endif // UTIL_ARCH_H
