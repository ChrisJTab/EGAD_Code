//
// Created by Shujian Qian on 2023-09-20.
//
/*
This file defines transaction bridges (TxnBridge and PackedTxnBridge) for efficient data transfers between CPU and GPU.
TxnBridge: Transfers standard transactions (TxnArray)
PackedTxnBridge: Transfers packed transactions (PackedTxnArray)
Both bridges allow transactions to move between different devices (CPU ↔ GPU)
*/

#ifndef TXN_BRIDGE_H
#define TXN_BRIDGE_H

#include <any>
#include <cstdint>
#include <cstdio>

#include "common.h"
#include "txn.h"
#include "util_device_type.h"
#include "util_exec_mode.h"
#include "util_gpu_transfer.h"

namespace epic {

/**
 * POD describing a single bulk-copy operation prepared by a TxnBridge.
 *
 * After `Link()` is called the bridge fills in:
 *   • `src_type` / `dst_type`   – CPU or GPU endpoint
 *   • `txn_size`                – bytes per transaction
 *   • `num_txns`                – total transactions to copy
 *   • `src_ptr` / `dest_ptr`    – base addresses of the contiguous blob
 *
 * It is **not used directly** by client code; it simply groups parameters
 * that the internal copy helpers consume.
 */
struct TxnBridgeStorage
{
    DeviceType src_type;
    DeviceType dst_type;
    size_t txn_size;
    size_t num_txns;
    uint8_t *src_ptr;
    uint8_t *dest_ptr;
};

/**
 * Thin convenience wrapper for copying a *plain* `TxnArray<T>` between devices.
 *
 * Typical life-cycle
 * ------------------
 *   1. `Link(src, dst)`  – remember pointers, decide CPU↔GPU direction,
 *                          create a CUDA stream if either side is on the GPU.
 *   2. `StartTransfer()` – enqueue an async memcpy (host↔device or device↔host).
 *   3. Do other work on the CPU while the copy runs.
 *   4. `FinishTransfer()` – sync the stream so the caller knows the data is visible.
 *
 * If `src.device == dst.device` the bridge just repoints `dst.txns`
 * to the existing buffer – no copy is issued.
 */
class TxnBridge
{
    DeviceType src_type = DeviceType::CPU;
    DeviceType dst_type = DeviceType::CPU;
    size_t txn_size = 0;
    size_t num_txns = 0;
    void *src_ptr = nullptr;
    void *dest_ptr = nullptr;
    std::any copy_stream; /* used for GPU only */
public:

    /**
     * Bind a **source** and **destination** `TxnArray<T>`.
     * Ensures both arrays are initialised, sets internal bookkeeping
     * fields and allocates a CUDA stream on first GPU use.
     *
     * Throws if the two arrays disagree on `num_txns`.
     */
    template<typename TxnType>
    void Link(TxnArray<TxnType> &src, TxnArray<TxnType> &dest)
    {
        auto &logger = Logger::GetInstance();
        if (src.num_txns != dest.num_txns)
        {
            logger.Error("TxnArray::num_txns mismatch");
        }

        src_type = src.device;
        dst_type = dest.device;
        txn_size = BaseTxnSize<TxnType>::value;
        num_txns = src.num_txns;

        if (src.txns == nullptr)
        {
            src.Initialize();
        }

        if (src.device == dest.device)
        {
            /* FIXME: add a relink function */
//            if (dest.txns != nullptr)
//            {
//                dest.Destroy();
//            }
            dest.txns = src.txns; // no need to copy, just point to the same memory
        }
        else
        {
            if (dest.txns == nullptr)
            {
                dest.Initialize();
            }
        }
        /*
        GPU operations are asynchronous, so a stream allows overlapping memory transfers with computation
        */
        if (src.device == DeviceType::GPU || dest.device == DeviceType::GPU)
        {
            if (!copy_stream.has_value())
            {
                copy_stream = createGpuStream();
                logger.Trace("Created a new stream for GPU copy with type {}", copy_stream.type().name());
            }
        }

        src_ptr = src.txns;
        dest_ptr = dest.txns;
    }

    ~TxnBridge()
    {
        if (copy_stream.has_value())
        {
            destroyGpuStream(copy_stream);
        }
    }

    /**
     * Kick off the bulk copy recorded in `Link()`.
     * CPU↔CPU            → nothing to do
     * GPU↔GPU (same dev) → nothing to do
     * CPU→GPU            → `cudaMemcpyAsync` on `copy_stream`
     * GPU→CPU            → same in the other direction
     */
    virtual void StartTransfer();

    /** Block until the async copy launched in `StartTransfer()` is finished. */
    virtual void FinishTransfer();
};


/**
 * Bridge variant for **PackedTxnArray<T>** where every epoch’s transactions
 * are stored as one variable-length blob plus an index[] table and a
 * `size` field that records the current packed byte-length.
 *
 * Responsibilities are the same as `TxnBridge`, but it must copy
 * *three* pieces of state atomically:
 *   • packed   byte blob  (`txns`)
 *   • per-txn  offset list (`index`)
 *   • current  packed size (`size`)
 *
 * When the endpoints are on the same device the bridge again aliases the
 * destination pointers to source memory instead of copying.
 */
class PackedTxnBridge
{
    DeviceType src_type = DeviceType::CPU;
    DeviceType dst_type = DeviceType::CPU;
    size_t txn_size = 0;
    size_t num_txns = 0;
    uint32_t packed_size = 0;
    void *src_ptr = nullptr;
    void *src_index_ptr = nullptr;
    void *dest_ptr = nullptr;
    void *dest_index_ptr = nullptr;
    uint32_t *src_size_ptr = nullptr;
    uint32_t *dest_size_ptr = nullptr;
    std::any copy_stream; /* used for GPU only */
public:

    /**
     * Bind a packed source and destination.  Verifies `num_txns`
     * match, allocates buffers if the destination is still empty,
     * sets internal pointers and initialises a CUDA stream on first use.
     */
    template<typename TxnType>
    void Link(PackedTxnArray<TxnType> &src, PackedTxnArray<TxnType> &dest)
    {
        auto &logger = Logger::GetInstance();
        if (src.num_txns != dest.num_txns)
        {
            throw std::runtime_error("TxnArray::num_txns mismatch");
        }

        src_type = src.device;
        dst_type = dest.device;
        txn_size = BaseTxnSize<TxnType>::value;
        num_txns = src.num_txns;

        if (src.txns == nullptr)
        {
            src.Initialize();
        }

        if (src.device == dest.device)
        {
            /* FIXME: add a relink function */
//            if (dest.txns != nullptr)
//            {
//                dest.Destroy();
//            }
            dest.txns = src.txns;
            dest.index = src.index;
        }
        else
        {
            if (dest.txns == nullptr)
            {
                dest.Initialize();
            }
        }

        if (src.device == DeviceType::GPU || dest.device == DeviceType::GPU)
        {
            if (!copy_stream.has_value())
            {
                copy_stream = createGpuStream();
                logger.Trace("Created a new stream for GPU copy with type {}", copy_stream.type().name());
            }
        }

        src_ptr = src.txns;
        dest_ptr = dest.txns;
        src_index_ptr = src.index;
        dest_index_ptr = dest.index;
        src_size_ptr = &src.size;
        dest_size_ptr = &dest.size;
    }

    virtual ~PackedTxnBridge()
    {
        if (copy_stream.has_value())
        {
            destroyGpuStream(copy_stream);
        }
    }

    /**
     * Enqueue the appropriate memcpy:
     *  - CPU→CPU or GPU→GPU (same dev): no-op
     *  - GPU→CPU: read back `size`, then copy `txns` and `index`
     *  - CPU→GPU: copy `txns` and `index`, propagate `size`
     */

    virtual void StartTransfer()
    {
        auto &logger = Logger::GetInstance();
        if (src_type == DeviceType::CPU && dst_type == DeviceType::CPU)
        {
            return;
        }
#ifdef EPIC_CUDA_AVAILABLE
        else if (src_type == DeviceType::GPU && dst_type == DeviceType::GPU)
        {
            return;
        }
        else if (src_type == DeviceType::GPU)
        {
            transferGpuToCpu(&packed_size, &static_cast<uint32_t *>(src_index_ptr)[num_txns], sizeof(uint32_t));
            transferGpuToCpu(dest_ptr, src_ptr, packed_size, copy_stream);
            transferGpuToCpu(dest_index_ptr, src_index_ptr, num_txns * sizeof(uint32_t), copy_stream);
            *src_size_ptr = packed_size;
            *dest_size_ptr = packed_size;
            logger.Info("Transferred {} bytes from GPU to CPU", packed_size);
        }
        else if (dst_type == DeviceType::GPU)
        {
            packed_size = *src_size_ptr;
            transferCpuToGpu(dest_ptr, src_ptr, packed_size, copy_stream);
            transferCpuToGpu(dest_index_ptr, src_index_ptr, num_txns * sizeof(uint32_t), copy_stream);
            *dest_size_ptr = packed_size;
        }
#endif
        else
        {
            logger.Error("Unsupported transfer using TxnBridge");
        }
    }

    /**
     * Synchronise the CUDA stream if a GPU was involved; otherwise no-op.
     */
    virtual void FinishTransfer()
    {
        if (src_type == DeviceType::CPU && dst_type == DeviceType::CPU)
        {
            return;
        }
#ifdef EPIC_CUDA_AVAILABLE
        else if (src_type == DeviceType::GPU && dst_type == DeviceType::GPU)
        {
            return;
        }
        else if (src_type == DeviceType::GPU || dst_type == DeviceType::GPU)
        {
            syncGpuStream(copy_stream);
        }
#endif
        else
        {
            auto &logger = Logger::GetInstance();
            logger.Error("Unsupported transfer using TxnBridge");
        }
    }
};
} // namespace epic

#endif // TXN_BRIDGE_H
