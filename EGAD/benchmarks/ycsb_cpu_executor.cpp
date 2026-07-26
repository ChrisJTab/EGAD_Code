//
// Created by Shujian Qian on 2023-12-05.
//

#include <benchmarks/ycsb_cpu_executor.h>

namespace epic::ycsb {

void CpuExecutor::execute(uint32_t epoch)
{
    uint32_t num_threads = config.cpu_exec_num_threads;
    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < num_threads; i++)
    {
        threads.emplace_back(&CpuExecutor::executionWorker, this, epoch, i);
    }

    for (auto &t : threads)
    {
        t.join();
    }
}

void CpuExecutor::executionWorker(uint32_t epoch, uint32_t thread_id)
{
#if 0
    uint32_t num_txns = config.num_txns;
    uint32_t num_pieces = num_txns * 10;
    uint32_t num_threads = config.cpu_exec_num_threads;
    for (uint32_t piece_id = thread_id; piece_id < num_pieces; piece_id += num_threads)
    {
        uint32_t txn_id = piece_id / 10;
        uint32_t piece_offset = piece_id % 10;
        YcsbTxnParam *txn_param = reinterpret_cast<YcsbTxnParam *>(txn.getTxn(txn_id)->data);
        YcsbExecPlan *exec_plan = reinterpret_cast<YcsbExecPlan *>(plan.getTxn(txn_id)->data);
        YcsbValue value;
        switch (txn_param->ops[piece_offset])
        {
        case YcsbOpType::FULL_READ: {
            readFromTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].read_plan.read_loc, epoch, &value);
            break;
        }
        case YcsbOpType::FULL_READ_MODIFY_WRITE:
        case YcsbOpType::READ_MODIFY_WRITE: {
            readFromTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].read_modify_write_plan.read_loc,
                epoch, &value);
            memset(&value.data[txn_param->field_ids[piece_offset]], txn_id, sizeof(value.data[0]));
            writeToTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].read_modify_write_plan.write_loc,
                epoch, &value);
            break;
        }
        case YcsbOpType::UPDATE: {
            readFromTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].copy_update_plan.read_loc, epoch,
                &value);
            memset(&value.data[txn_param->field_ids[piece_offset]], txn_id, sizeof(value.data[0]));
            writeToTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].copy_update_plan.write_loc, epoch,
                &value);
            break;
        }
        default:
            throw std::runtime_error("epic::ycsb::CpuExecutor::executionWorker() found unknown op type.");
        }
        txn_param->record_ids[piece_offset] =
            value.data[txn_param->field_ids[piece_offset]][10]; /* prevent optimization */
    }
#else
        uint32_t num_txns = config.num_txns;
        uint32_t num_pieces = num_txns * 10;
        uint32_t num_threads = config.cpu_exec_num_threads;
        for (uint32_t txn_id = thread_id; txn_id < num_txns; txn_id += num_threads)
        {
            for (uint32_t piece_offset = 0; piece_offset < 10; piece_offset++)
            {

                YcsbTxnParam *txn_param = reinterpret_cast<YcsbTxnParam *>(txn.getTxn(txn_id)->data);
                YcsbExecPlan *exec_plan = reinterpret_cast<YcsbExecPlan *>(plan.getTxn(txn_id)->data);

                if (config.split_field)
                {
                    /* ---- split-field path: each field is an independent Record<YcsbFieldValue> ---- */
                    YcsbFieldRecords *sf_records = std::get<YcsbFieldRecords *>(records);
                    YcsbFieldVersions *sf_versions = std::get<YcsbFieldVersions *>(versions);
                    YcsbFieldValue field_value;
                    uint32_t phys_id = txn_param->record_ids[piece_offset] * 10
                                       + txn_param->field_ids[piece_offset];

                    switch (txn_param->ops[piece_offset])
                    {
                    case YcsbOpType::READ: {
                        readFromTable(sf_records, sf_versions, phys_id,
                            exec_plan->plans[piece_offset].read_plan.read_loc, epoch, &field_value);
                        break;
                    }
                    case YcsbOpType::FULL_READ: {
                        uint32_t base_id = txn_param->record_ids[piece_offset] * 10;
                        for (uint32_t j = 0; j < 10; ++j)
                        {
                            readFromTable(sf_records, sf_versions, base_id + j,
                                exec_plan->plans[piece_offset].full_read_plan.read_locs[j],
                                epoch, &field_value);
                        }
                        break;
                    }
                    case YcsbOpType::READ_MODIFY_WRITE: {
                        readFromTable(sf_records, sf_versions, phys_id,
                            exec_plan->plans[piece_offset].read_modify_write_plan.read_loc,
                            epoch, &field_value);
                        memset(&field_value.data[0], txn_id, sizeof(field_value.data));
                        writeToTable(sf_records, sf_versions, phys_id,
                            exec_plan->plans[piece_offset].read_modify_write_plan.write_loc,
                            epoch, &field_value);
                        break;
                    }
                    case YcsbOpType::FULL_READ_MODIFY_WRITE: {
                        uint32_t base_id = txn_param->record_ids[piece_offset] * 10;
                        for (uint32_t j = 0; j < 10; ++j)
                        {
                            readFromTable(sf_records, sf_versions, base_id + j,
                                exec_plan->plans[piece_offset].full_read_modify_write_plan.read_locs[j],
                                epoch, &field_value);
                        }
                        /* Re-read the target field to stage a correct modification buffer.
                           readFromTable is idempotent once the version is stamped. */
                        readFromTable(sf_records, sf_versions, phys_id,
                            exec_plan->plans[piece_offset].full_read_modify_write_plan.read_locs[txn_param->field_ids[piece_offset]],
                            epoch, &field_value);
                        memset(&field_value.data[0], txn_id, sizeof(field_value.data));
                        writeToTable(sf_records, sf_versions, phys_id,
                            exec_plan->plans[piece_offset].full_read_modify_write_plan.write_loc,
                            epoch, &field_value);
                        break;
                    }
                    case YcsbOpType::UPDATE: {
                        /* split-field UPDATE is write-only (matches GPU submitter behavior). */
                        memset(&field_value.data[0], txn_id, sizeof(field_value.data));
                        writeToTable(sf_records, sf_versions, phys_id,
                            exec_plan->plans[piece_offset].update_plan.write_loc,
                            epoch, &field_value);
                        break;
                    }
                    case YcsbOpType::INSERT: {
                        uint32_t base_id = txn_param->record_ids[piece_offset] * 10;
                        for (uint32_t j = 0; j < 10; ++j)
                        {
                            memset(&field_value.data[0], txn_id, sizeof(field_value.data));
                            writeToTable(sf_records, sf_versions, base_id + j,
                                exec_plan->plans[piece_offset].insert_full_plan.write_locs[j],
                                epoch, &field_value);
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error(
                            "epic::ycsb::CpuExecutor::executionWorker() split-field: unknown op type.");
                    }
                    /* prevent the compiler from optimizing the reads away */
                    txn_param->record_ids[piece_offset] = field_value.data[10];
                }
                else
                {
                    /* ---- original non-split path ---- */
                    YcsbValue value;
                    switch (txn_param->ops[piece_offset])
                    {
                    case YcsbOpType::READ:
                    case YcsbOpType::FULL_READ: {
                        readFromTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                            txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].read_plan.read_loc, epoch,
                            &value);
                        break;
                    }
                    case YcsbOpType::FULL_READ_MODIFY_WRITE:
                    case YcsbOpType::READ_MODIFY_WRITE: {
                        readFromTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                            txn_param->record_ids[piece_offset],
                            exec_plan->plans[piece_offset].read_modify_write_plan.read_loc, epoch, &value);
                        memset(&value.data[txn_param->field_ids[piece_offset]], txn_id, sizeof(value.data[0]));
                        writeToTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                            txn_param->record_ids[piece_offset],
                            exec_plan->plans[piece_offset].read_modify_write_plan.write_loc, epoch, &value);
                        break;
                    }
                    case YcsbOpType::UPDATE: {
                        readFromTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                            txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].copy_update_plan.read_loc,
                            epoch, &value);
                        memset(&value.data[txn_param->field_ids[piece_offset]], txn_id, sizeof(value.data[0]));
                        writeToTable(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                            txn_param->record_ids[piece_offset], exec_plan->plans[piece_offset].copy_update_plan.write_loc,
                            epoch, &value);
                        break;
                    }
                    default:
                        throw std::runtime_error("epic::ycsb::CpuExecutor::executionWorker() found unknown op type.");
                    }
                    txn_param->record_ids[piece_offset] =
                        value.data[txn_param->field_ids[piece_offset]][10]; /* prevent optimization */
                }
            }
        }
#endif
}

} // namespace epic::ycsb
