//
// Created by Shujian Qian on 2023-08-09.
//

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <cstdlib>

namespace epic {
class Allocator
{
public:
    virtual void *Allocate(size_t size) = 0;
    virtual void Free(void *ptr) = 0;
    virtual void PrintMemoryInfo() = 0;
    virtual void PrintMemoryInfoCpu() = 0;
    virtual void GetMemoryInfo(size_t &free, size_t &total) = 0;
    virtual void GetMemoryInfoCpu(size_t &free, size_t &total) = 0;
};
} // namespace epic

#endif // ALLOCATOR_H
