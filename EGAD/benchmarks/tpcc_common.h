//
// Created by Shujian Qian on 2023-08-29.
//

#ifndef TPCC_COMMON_H
#define TPCC_COMMON_H

#include <cstdlib>
#include <cstdint>

namespace epic::tpcc {

// kMaxWarehouses sets the bitfield width for w_id-shaped fields via
// ceilLog2(2 * kMaxWarehouses). At 64, the bitfield was 7 bits and
// W=128 silently wrapped w_id=128 to 0 (cuco tolerated this because
// no other key wrapped to the same value, but the flat-array OL index
// reads the bitfield as a numeric value and would compute an OOB
// index). Bumped to 128 to give 8 bits, keeping the underlying uint64_t
// packed-key type unchanged.
constexpr size_t kMaxWarehouses = 128;
constexpr uint32_t kMaxCustomers = 3000;

} // namespace epic::tpcc

#endif // TPCC_COMMON_H
