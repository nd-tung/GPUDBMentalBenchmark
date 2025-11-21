#pragma once
#include <vector>
#include <string>
#include <map>
#include <cstdint>

namespace engine {

struct GroupByResult {
    std::map<uint32_t, double> groups;  // key -> aggregated value
    double gpu_ms;
    double upload_ms;
};

// GPU-based GROUP BY executor with hash aggregation
class GroupByExecutor {
public:
    // Check if query can use GPU group by
    static bool isEligible(const std::string& groupByColumn, const std::string& aggColumn);
    
    // Execute GPU GROUP BY with SUM aggregation
    // groupByColumn: the column to group by (must be integer/key column)
    // aggColumn: the column to aggregate (SUM)
    static GroupByResult runGroupBySum(const std::string& dataset_path,
                                       const std::string& table,
                                       const std::string& groupByColumn,
                                       const std::string& aggColumn);
};

} // namespace engine
