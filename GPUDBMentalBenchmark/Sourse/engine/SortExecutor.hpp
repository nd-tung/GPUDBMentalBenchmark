#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace engine {

struct SortResult {
    std::vector<uint32_t> indices;  // Sorted row indices
    double gpu_ms;
    double upload_ms;
};

// GPU-based ORDER BY executor using bitonic sort
class SortExecutor {
public:
    // Check if query can use GPU sorting
    static bool isEligible(const std::string& orderByColumn, uint32_t rowCount);
    
    // Execute GPU sort on a single column
    // Returns sorted indices that can be used to reorder result rows
    static SortResult runSort(const std::string& dataset_path,
                              const std::string& table,
                              const std::string& orderByColumn,
                              bool ascending = true);
};

} // namespace engine
