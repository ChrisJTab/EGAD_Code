//
// Created by Christian Tabbah on 2025-09-16.
//

#ifndef UTIL_EXEC_MODE_H
#define UTIL_EXEC_MODE_H

#include <cstdint>

namespace epic {

enum class ExecMode : uint32_t
{
    CPU_ONLY = 0,
    GPU_ONLY,
    HYBRID_STAGING
};

}

#endif // UTIL_EXEC_MODE_H
