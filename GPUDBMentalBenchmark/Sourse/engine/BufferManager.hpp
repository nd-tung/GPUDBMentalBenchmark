#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace engine {

class BufferManager {
public:
    void* allocate(std::size_t bytes);
    void deallocate(void* ptr);
};

} // namespace engine
