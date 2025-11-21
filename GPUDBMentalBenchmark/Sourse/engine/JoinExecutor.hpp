#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace engine {

struct JoinResult {
    double revenue = 0.0;
    double gpu_ms = 0.0;
    double upload_ms = 0.0;
    uint32_t match_count = 0;
};

class JoinExecutor {
public:
    // Execute hash join between two tables on GPU
    // Returns aggregated result after join
    static JoinResult runHashJoin(
        const std::string& dataset_path,
        const std::string& leftTable,
        const std::string& rightTable,
        const std::string& leftKeyColumn,
        const std::string& rightKeyColumn,
        const std::string& aggColumn,
        const std::vector<std::string>& predicateColumns = {}
    );
    
    // Check if join is eligible for GPU execution
    static bool isEligible(const std::string& leftTable,
                          const std::string& rightTable);
};

} // namespace engine
