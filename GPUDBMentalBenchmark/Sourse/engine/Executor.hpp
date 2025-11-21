#pragma once
#include <string>
#include "IR.hpp"
#include "PipelineBuilder.hpp"
#include "Operators.hpp"

namespace engine {

class Executor {
public:
    struct Result { double revenue; double cpu_ms; };
    static Result runQ6(const PipelineSpecQ6& spec, const std::string& dataset_path);
    static Result runGeneric(const Plan& plan, const std::string& dataset_path);
};

} // namespace engine