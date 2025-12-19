#pragma once
#include <vector>
#include <string>
#include <map>
#include <cstdint>

namespace engine {

struct GroupByResult {
    std::map<std::vector<uint32_t>, std::vector<double>> groups;  // composite key -> multiple aggregated values
    double gpu_ms;
    double upload_ms;
};

// GPU-based GROUP BY executor with hash aggregation
class GroupByExecutor {
public:
    // Check if query can use GPU group by
    static bool isEligible(const std::vector<std::string>& groupByColumns, 
                          const std::vector<std::string>& aggColumns);
    
    // Execute GPU GROUP BY with multiple keys and multiple aggregates
    // groupByColumns: the columns to group by (can be multiple)
    // aggColumns: the columns to aggregate
    // aggFuncs: aggregate functions (SUM, AVG, MIN, MAX, COUNT)
    static GroupByResult runGroupBy(const std::string& dataset_path,
                                    const std::string& table,
                                    const std::vector<std::string>& groupByColumns,
                                    const std::vector<std::string>& aggColumns,
                                    const std::vector<std::string>& aggFuncs);
    
    // Legacy single-column interface for backwards compatibility
    static GroupByResult runGroupBySum(const std::string& dataset_path,
                                       const std::string& table,
                                       const std::string& groupByColumn,
                                       const std::string& aggColumn);
};

} // namespace engine
