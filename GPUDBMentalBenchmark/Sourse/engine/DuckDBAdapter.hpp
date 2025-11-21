#pragma once
#include <string>

namespace engine {

class DuckDBAdapter {
public:
    // Returns EXPLAIN (FORMAT JSON) output as a JSON array string, or empty on failure.
    static std::string explainJSON(const std::string& sql);
    // Runs the query and returns the first numeric cell as double (for Q6 validation), NaN on failure.
    static double runScalarDouble(const std::string& sql);
};

} // namespace engine
