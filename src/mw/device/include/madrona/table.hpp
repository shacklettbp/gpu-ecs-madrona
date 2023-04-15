#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include <madrona/sync.hpp>
#include <madrona/types.hpp>

namespace madrona {

struct TypeInfo {
    uint32_t alignment;
    uint32_t numBytes;
};

struct Table {
    static constexpr uint32_t maxColumns = 32;

    std::array<void *, maxColumns> columns;

    // FIXME: move a lot of this metadata out of the core table struct
    std::array<uint32_t, maxColumns> columnSizes;
    int32_t numColumns;

    int32_t numRows;
    int32_t numAllocatedRows;
};

}
