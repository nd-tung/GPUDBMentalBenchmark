#include "HashTable.hpp"

namespace engine {

void HashTableU32::allocate(std::size_t capacity) {
    capacity_ = capacity;
    keys_.assign(capacity_, 0);
    vals_.assign(capacity_, 0);
}

void HashTableU32::clear() {
    std::fill(keys_.begin(), keys_.end(), 0);
    std::fill(vals_.begin(), vals_.end(), 0);
}

} // namespace engine
