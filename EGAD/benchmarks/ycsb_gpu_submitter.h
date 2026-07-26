//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_GPU_SUBMITTER_H
#define EPIC_BENCHMARKS_YCSB_GPU_SUBMITTER_H

#include <benchmarks/ycsb_submitter.h>

namespace epic::ycsb {

class YcsbGpuSubmitter : public YcsbSubmitter
{
public:
    YcsbGpuSubmitter(TableSubmitDest submit_dest, YcsbConfig config);
    ~YcsbGpuSubmitter();

    void submit(TxnArray<YcsbTxnParam> &txn_array) override;

private:
    // Pre-allocated device counters (avoid cudaMalloc/cudaFree per submit)
    uint32_t* d_read_count_  = nullptr;
    uint32_t* d_write_count_ = nullptr;
    uint32_t* d_insert_count_ = nullptr;
    // Mapped host memory for reading back scalars without D2H copy engine
    // [0]=num_ops, [1]=read_count, [2]=write_count, [3]=insert_count
    uint32_t* h_mapped_ = nullptr;
    uint32_t* d_mapped_num_ops_  = nullptr; // device pointer into mapped[0]
    uint32_t* d_mapped_read_cnt_ = nullptr; // device pointer into mapped[1]
    uint32_t* d_mapped_write_cnt_ = nullptr; // device pointer into mapped[2]
    uint32_t* d_mapped_insert_cnt_ = nullptr; // device pointer into mapped[3]
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_GPU_SUBMITTER_H
