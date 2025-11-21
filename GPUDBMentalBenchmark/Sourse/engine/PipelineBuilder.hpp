#pragma once
#include <string>
#include "IR.hpp"
#include "Operators.hpp"

namespace engine {

struct PipelineSpecQ6 {
    FilterQ6::Params params;
};

class PipelineBuilder {
public:
    // For MVP, recognize Q6-like plans and construct FilterQ6 params.
    static PipelineSpecQ6 buildQ6(const Plan& plan);
};

} // namespace engine