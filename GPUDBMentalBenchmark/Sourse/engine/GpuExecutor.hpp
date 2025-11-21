#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "Predicate.hpp"

namespace engine {

struct GPUResult { double revenue; double gpu_ms; double upload_ms; uint64_t count; };

// GPU executor: supports COUNT, SUM, AVG, MIN, MAX with predicates and expressions.
class GpuExecutor {
public:
    static bool isEligible(const std::string& aggFunc,
                           const std::vector<expr::Clause>& clauses,
                           const std::string& targetColumn);

    static GPUResult runSum(const std::string& dataset_path,
                            const std::string& targetColumn,
                            const std::vector<expr::Clause>& clauses);
    
    // Run SUM with arithmetic expression (e.g., l_extendedprice * (1 - l_discount))
    static GPUResult runSumWithExpression(const std::string& dataset_path,
                                          const std::string& expression,
                                          const std::vector<expr::Clause>& clauses);
    
    // Generic aggregation: COUNT, AVG, MIN, MAX
    static GPUResult runAggregate(const std::string& dataset_path,
                                  const std::string& aggFunc,
                                  const std::string& targetColumn,
                                  const std::vector<expr::Clause>& clauses);
};

} // namespace engine
