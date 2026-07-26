//
// Created by Shujian Qian on 2023-11-20.
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <cuda_runtime.h>

#include "util_memory.h"

namespace epic {

// Pinned host memory uses malloc + cudaHostRegister rather
// than cudaMallocHost, so page ownership stays with the host allocator
// rather than the CUDA runtime; cudaHostRegister pins the pages for DMA.
// Steady-state DMA performance is identical (the buffer is pinned either
// way).
void *allocatePinnedMemory(size_t size)
{
    void *retval = malloc(size);
    if (retval) {
        cudaError_t err = cudaHostRegister(retval, size, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "[allocatePinnedMemory] cudaHostRegister(%p, %zu) failed: %s; using pageable\n",
                    retval, size, cudaGetErrorString(err));
        }
    }
    return retval;
}

// See util_memory.h. EPIC_RECOVER_FROM (re-map existing,
// preserve bytes) takes precedence over EPIC_DURABLE_STORE (create + zero);
// neither set -> Malloc (byte-identical normal path). On any failure, logs and
// falls back to Malloc so a misconfigured durable run still completes.
void *MallocDurable(const char *name, size_t size, bool pin)
{
    const char *recover_dir = std::getenv("EPIC_RECOVER_FROM");
    const char *create_dir  = std::getenv("EPIC_DURABLE_STORE");
    const bool recover = (recover_dir != nullptr);
    const char *dir = recover ? recover_dir : create_dir;
    if (!dir) {
        return Malloc(size);  // normal path: pinned + zeroed, unchanged
    }
    std::string path = std::string(dir) + "/" + name + ".bin";
    int fd = open(path.c_str(), recover ? O_RDWR : (O_RDWR | O_CREAT), 0600);
    if (fd < 0) {
        fprintf(stderr, "[MallocDurable] open(%s) failed: %s; falling back to Malloc\n",
                path.c_str(), std::strerror(errno));
        return Malloc(size);
    }
    if (!recover && ftruncate(fd, static_cast<off_t>(size)) != 0) {
        fprintf(stderr, "[MallocDurable] ftruncate(%s, %zu) failed: %s\n",
                path.c_str(), size, std::strerror(errno));
    }
    void *retval = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);  // the mapping keeps the file alive
    if (retval == MAP_FAILED) {
        fprintf(stderr, "[MallocDurable] mmap(%s, %zu) failed: %s; falling back to Malloc\n",
                path.c_str(), size, std::strerror(errno));
        return Malloc(size);
    }
    // Pin only when asked. The CPU Primary Store is never a cudaMemcpy
    // endpoint (DMA goes through separate pinned staging buffers), so pinning
    // buys no DMA speed here. The growing tables pass pin=false and instead get
    // eager page commit from the provision branch below (the ship config: the
    // same page residency pinning gave, without the inert DMA pin).
    bool provisioned = false;
    if (pin) {
        cudaError_t err = cudaHostRegister(retval, size, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "[MallocDurable] cudaHostRegister(%p, %zu) failed: %s; using pageable mmap\n",
                    retval, size, cudaGetErrorString(err));
        }
    } else if (!recover) {
        // Provision the durable store on create: eagerly commit every page now.
        // This is the ship config -- lazy first-touch commit stalls mid-run and
        // loses to the CPU baseline, so provisioning is unconditional. Never on
        // recover, which must preserve the existing durable bytes (a memset would
        // wipe the crashed run's state we are reloading).
        std::memset(retval, 0, size);
        provisioned = true;
    }
    fprintf(stderr, "[MallocDurable] %s %s (%zu bytes, %s) at %p\n",
            recover ? "RECOVER-mapped" : "created", path.c_str(), size,
            pin ? "pinned" : (provisioned ? "provisioned" : "pageable"), retval);
    return retval;
}

// See util_memory.h. A real illegal device access poisons the
// CUDA context (cudaErrorIllegalAddress); we report it and terminate non-zero so
// the supervisor restarts a fresh process (which gets a clean context -- the
// in-process err-700 recovery that cudaDeviceReset cannot do). _exit (not
// std::exit) skips CUDA atexit handlers against the poisoned context; the
// MAP_SHARED durable store is already in the page cache, so nothing needs
// flushing. Process death also frees the poisoned GPU context for the next run.
#ifdef EGAD_VALIDATION
__global__ void k_inject_illegal_access()
{
    volatile int *p = reinterpret_cast<int *>(0xdeadbeef00ULL);
    *p = 0xBAD;
}
void injectGpuFault()
{
    k_inject_illegal_access<<<1, 1>>>();
    cudaError_t e = cudaDeviceSynchronize();
    fprintf(stderr, "[CRASH] injected GPU fault: cudaDeviceSynchronize -> %d (%s)\n",
            static_cast<int>(e), cudaGetErrorString(e));
    fflush(nullptr);
    _exit(e != cudaSuccess ? static_cast<int>(e) : 1);
}
#endif

void freePinedMemory(void *ptr)
{
    if (ptr) {
        // cudaHostUnregister returns cudaErrorHostMemoryNotRegistered if
        // the matching cudaHostRegister failed earlier; safe to ignore.
        cudaHostUnregister(ptr);
    }
    free(ptr);
}

} // namespace epic