#include "BufferManager.hpp"
#include <cstdlib>

namespace engine {

void* BufferManager::allocate(std::size_t bytes) {
    return std::malloc(bytes);
}

void BufferManager::deallocate(void* ptr) {
    std::free(ptr);
}

} // namespace engine
