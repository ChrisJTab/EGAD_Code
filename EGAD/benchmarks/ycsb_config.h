//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_CONFIG_H
#define EPIC_BENCHMARKS_YCSB_CONFIG_H

#include <cstdint>
#include <cstddef>
#include <cassert>

#include <util_device_type.h>
#include <util_exec_mode.h>

namespace epic::ycsb {

struct YcsbConfig
{
    static constexpr size_t num_ops_per_txn = 10;
    /*
    Example: If we defined txn_mix = {50, 50, 0, 0}, then the transaction mix would be: 50% reads, 50% writes, 0% rmws, 0% inserts
    rmws are read-modify-writes
    */

    struct YcsbTxnMix
    {
        size_t num_reads = num_ops_per_txn;
        size_t num_writes = 0;
        size_t num_rmw = 0;
        size_t num_inserts = 0;
        size_t num_deletes = 0;

        YcsbTxnMix() = default;
        YcsbTxnMix(size_t num_reads, size_t num_writes, size_t num_rmw, size_t num_inserts)
            : num_reads(num_reads)
            , num_writes(num_writes)
            , num_rmw(num_rmw)
            , num_inserts(num_inserts)
        {
            assert(num_reads + num_writes + num_rmw + num_inserts == 100);
        }
        YcsbTxnMix(size_t num_reads, size_t num_writes, size_t num_rmw, size_t num_inserts, size_t num_deletes)
            : num_reads(num_reads)
            , num_writes(num_writes)
            , num_rmw(num_rmw)
            , num_inserts(num_inserts)
            , num_deletes(num_deletes)
        {
            assert(num_reads + num_writes + num_rmw + num_inserts + num_deletes == 100);
        }
    } txn_mix;
    size_t num_records = 2'500'000;
    size_t num_txns = 100'000;
    size_t starting_num_records = 2'500'000;
    size_t epochs = 20;
    double skew_factor = 0.0;
    DeviceType index_device = DeviceType::GPU;
    DeviceType initialize_device = DeviceType::GPU;
    DeviceType execution_device = DeviceType::GPU;
    ExecMode execution_mode = ExecMode::GPU_ONLY;
    bool split_field = true;
    bool full_record_read = true;
    uint32_t cpu_exec_num_threads = 1;
    uint32_t gpu_capacity = 0;
    bool overlap_flush = false;

    // Read bias toward recently-inserted CRIDs. With probability
    // recent_read_bias, a non-insert op draws its key uniformly from
    // [max_existing_record - recent_window_size, max_existing_record)
    // instead of the global Zipfian distribution; the remaining (1 -
    // recent_read_bias) of non-insert ops use the existing Zipfian path.
    // Mirrors TPC-C's Delivery-reads-recent-NewOrder-OrderLines pattern,
    // which is the cache-retention behaviour the EGAD insert prototype is
    // measured against. A value of 0.0 disables the bias.
    double recent_read_bias = 0.0;
    uint32_t recent_window_size = 0;
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_CONFIG_H
