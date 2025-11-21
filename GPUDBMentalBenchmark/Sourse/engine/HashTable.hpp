#pragma once
#include <cstdint>
#include <vector>

namespace engine {

class HashTableU32 {
public:
    explicit HashTableU32(std::size_t capacity = 0) { allocate(capacity); }
    void allocate(std::size_t capacity);
    void clear();
    std::size_t capacity() const { return capacity_; }

private:
    std::size_t capacity_{};
    std::vector<uint32_t> keys_;
    std::vector<uint32_t> vals_;
};

} // namespace engine
