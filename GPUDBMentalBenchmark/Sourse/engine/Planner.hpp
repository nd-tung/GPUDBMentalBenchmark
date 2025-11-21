#pragma once
#include <string>
#include "IR.hpp"

namespace engine {

class Planner {
public:
    // Very small MVP: build a linear plan for Q6-like query using DuckDB EXPLAIN JSON text.
    static Plan fromSQL(const std::string& sql);
};

} // namespace engine
