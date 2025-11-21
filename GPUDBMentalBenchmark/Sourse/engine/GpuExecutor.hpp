#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "Predicate.hpp"

namespace engine {

struct GPUResult { double revenue; double gpu_ms; double upload_ms; };

// Minimal GPU executor: supports SUM(single float column) with simple predicates.
class GpuExecutor {
public:
    static bool isEligible(const std::string& aggFunc,
                           const std::vector<expr::Clause>& clauses,
                           const std::string& targetColumn);

    static GPUResult runSum(const std::string& dataset_path,
                            const std::string& targetColumn,
                            const std::vector<expr::Clause>& clauses);
};

} // namespace engine
