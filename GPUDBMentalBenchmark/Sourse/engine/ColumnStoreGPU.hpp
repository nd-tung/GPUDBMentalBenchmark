// Minimal GPU column staging (initial version)
#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>

// Forward declare Metal types (included in .cpp)
namespace MTL { class Device; class Buffer; class Library; class CommandQueue; }

namespace engine {

struct GPUColumn {
    std::string name;
    std::size_t count = 0;
    MTL::Buffer* buffer = nullptr; // Shared memory buffer
};

// Simple singleton staging cache. For now only float columns are supported.
class ColumnStoreGPU {
public:
    static ColumnStoreGPU& instance();

    void initialize(); // lazy Metal device/library acquisition

    // Upload (or reuse) a float column. Returns GPUColumn* (owned by store).
    GPUColumn* stageFloatColumn(const std::string& name,
                                const std::vector<float>& data);

    MTL::Device* device() const { return m_device; }
    MTL::Library* library() const { return m_library; }
    MTL::CommandQueue* queue() const { return m_queue; }

private:
    ColumnStoreGPU() = default;
    ColumnStoreGPU(const ColumnStoreGPU&) = delete;
    ColumnStoreGPU& operator=(const ColumnStoreGPU&) = delete;

    MTL::Device* m_device = nullptr;
    MTL::Library* m_library = nullptr;
    MTL::CommandQueue* m_queue = nullptr;
    std::map<std::string, GPUColumn> m_columns; // simple name→column
};

} // namespace engine
