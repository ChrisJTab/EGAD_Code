//
// Created by Shujian Qian on 2023-08-09.
//

#include "gpu_allocator.h"

#include "util_gpu_error_check.cuh"
#include "util_log.h"
#include "util_math.h"
#include <cstdio>
#include <cstring>
#if defined(__linux__)
  #include <sys/sysinfo.h>
#endif

// System-wide RAM (same semantics as `free -h`)
static inline void GetMemoryInfoCpu(size_t& free_bytes, size_t& total_bytes) {
    struct sysinfo si{};
    if (sysinfo(&si) != 0) { free_bytes = total_bytes = 0; return; }
    // On Linux, sysinfo reports in si.mem_unit bytes on older kernels; newer kernels report mem_unit=1.
    const uint64_t unit = si.mem_unit ? si.mem_unit : 1;
    total_bytes = (size_t)(si.totalram * unit);
    // "free" (really: free pages); if you want closer to "MemAvailable", parse /proc/meminfo (below).
    free_bytes  = (size_t)(si.freeram  * unit);
}

// Optional: read MemAvailable (closer to what `free -h` shows as "available")
static inline size_t ReadMemAvailable() {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char key[64]; unsigned long long val_kib; char unit[16];
    size_t available = 0, total = 0;
    while (std::fscanf(f, "%63s %llu %15s", key, &val_kib, unit) == 3) {
        if (!std::strcmp(key, "MemAvailable:")) available = (size_t)val_kib * 1024ULL;
        if (!std::strcmp(key, "MemTotal:"))     total     = (size_t)val_kib * 1024ULL;
    }
    std::fclose(f);
    return available; // bytes
}

// Optional: per-process RSS (like top/ps)
static inline size_t GetProcessRSSBytes() {
    long rss_pages = 0;
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long pagesize = sysconf(_SC_PAGESIZE);
    long dummy;
    if (std::fscanf(f, "%ld %ld", &dummy, &rss_pages) != 2) { std::fclose(f); return 0; }
    std::fclose(f);
    return (size_t)rss_pages * (size_t)pagesize;
}

namespace epic {
void* GpuAllocator::Allocate(size_t size) {
    void* ptr = nullptr;

    auto s1 = cudaMalloc(&ptr, size);
    //fprintf(stderr, "[Alloc] cudaMalloc(%zu) -> %d, ptr=%p\n", size, (int)s1, ptr);
    if (s1 != cudaSuccess) { abort(); }

    auto s2 = cudaMemset(ptr, 0, size);
    //fprintf(stderr, "[Alloc] cudaMemset -> %d\n", (int)s2);
    if (s2 != cudaSuccess) { abort(); }

    return ptr;
}

void GpuAllocator::Free(void *ptr)
{
    gpu_err_check(cudaFree(ptr));
}

void GpuAllocator::PrintMemoryInfo()
{
    size_t free, total;
    gpu_err_check(cudaMemGetInfo(&free, &total));
    auto &logger = Logger::GetInstance();
    logger.Info("GPU memory usage: {} / {}", formatSizeBytes(total - free), formatSizeBytes(total));
}

void GpuAllocator::PrintMemoryInfoCpu()
{
    size_t sys_free, sys_total; GetMemoryInfoCpu(sys_free, sys_total);
    size_t mem_avail = ReadMemAvailable();
    size_t rss = GetProcessRSSBytes();
    auto &logger = Logger::GetInstance();
    logger.Info("CPU mem (sysinfo): free={} / total={}", formatSizeBytes(sys_free), formatSizeBytes(sys_total));
    logger.Info("CPU mem (MemAvailable): {}", formatSizeBytes(mem_avail));
    logger.Info("Process RSS: {}", formatSizeBytes(rss));
}

// also create a function for CPU memory ussage
void GpuAllocator::GetMemoryInfoCpu(size_t &free, size_t &total)
{
    // Prefer /proc/meminfo → MemAvailable/ MemTotal (kB)
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char key[64]; unsigned long long val_kb; char unit[32];
        unsigned long long mem_total_kb = 0, mem_available_kb = 0;

        while (std::fscanf(f, "%63s %llu %31s", key, &val_kb, unit) == 3) {
            if (std::strcmp(key, "MemTotal:") == 0)       mem_total_kb = val_kb;
            else if (std::strcmp(key, "MemAvailable:") == 0) mem_available_kb = val_kb;
        }
        std::fclose(f);

        if (mem_total_kb && mem_available_kb) {
            total = static_cast<size_t>(mem_total_kb) * 1024ull;
            free  = static_cast<size_t>(mem_available_kb) * 1024ull;
            return;
        }
    }

    // Fallback: sysinfo (approximate “available” as free+buffers)
    struct sysinfo info{};
    if (sysinfo(&info) == 0) {
        unsigned long long unit = info.mem_unit ? info.mem_unit : 1ull;
        total = static_cast<size_t>(info.totalram) * unit;
        unsigned long long avail = (unsigned long long)info.freeram + (unsigned long long)info.bufferram;
        free  = static_cast<size_t>(avail) * unit;
        return;
    }

    // Last resort
    free = total = 0;
}


// create a function to return free and total memory
void GpuAllocator::GetMemoryInfo(size_t &free, size_t &total)
{
    gpu_err_check(cudaMemGetInfo(&free, &total));
}


} // namespace epic