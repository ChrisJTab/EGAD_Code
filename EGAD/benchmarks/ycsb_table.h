//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_TABLE_H
#define EPIC_BENCHMARKS_YCSB_TABLE_H

#include <cstdint>

namespace epic::ycsb {

using YcsbKey = uint32_t;

// Record size selection. Default is the YCSB standard (10 fields × 100 B = 1000 B
// total), matching the original Cooper et al. SoCC'10 paper and EPIC OSDI'24.
// Define YCSB_SMALL_RECORDS to switch to a "small records" variant inspired by
// Silo SOSP'13 / Cicada SIGMOD'17 (~100 B records). We use 10 fields × 12 B =
// 120 B total; 12 B per field keeps sizeof(ValueType) divisible by 4 so the
// hardware-optimized 32-bit scatter_gather copy path works unchanged. Both
// regimes (~100 B vs 1000 B) are well-established in the OLTP research
// literature; exact byte count within each regime is not load-bearing.
#ifdef YCSB_SMALL_RECORDS
struct YcsbValue
{
    uint8_t data[10][12];
};

struct YcsbFieldValue
{
    uint8_t data[12];
};
#else
struct YcsbValue
{
    uint8_t data[10][100];
};

struct YcsbFieldValue
{
    uint8_t data[100];
};
#endif

}

#endif // EPIC_BENCHMARKS_YCSB_TABLE_H
