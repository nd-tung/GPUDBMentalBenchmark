#include "DuckDBAdapter.hpp"
#include <cstdio>
#include <array>
#include <memory>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>

namespace engine {

static std::string run_cmd_capture(const std::string& cmd) {
    std::array<char, 4096> buf{};
    std::string out;
#if defined(__APPLE__)
    FILE* pipe = popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return out;
    while (fgets(buf.data(), buf.size(), pipe)) { out.append(buf.data()); }
    pclose(pipe);
    return out;
}

std::string DuckDBAdapter::explainJSON(const std::string& sql) {
    // Use :memory:, load schema, then EXPLAIN JSON. Suppress stderr to avoid noisy errors.
    std::ostringstream oss;
    oss << "duckdb :memory: "
        << "-c \".read ../schema.sql\" "
        << "-c \"EXPLAIN (FORMAT JSON) " << sql << ";\" 2>/dev/null";
    return run_cmd_capture(oss.str());
}

double DuckDBAdapter::runScalarDouble(const std::string& sql) {
    std::ostringstream oss;
    // Note: running against :memory: without data will not produce actual values.
    // This is kept for future use when populating :memory: with COPY ...
    oss << "duckdb :memory: -c \".read ../schema.sql\" -c \"" << sql << ";\" 2>/dev/null";
    std::string out = run_cmd_capture(oss.str());
    // Try to find the last numeric token in the output
    // DuckDB prints a header row by default; we keep it simple for MVP by scanning tokens.
    double val = std::nan("");
    std::istringstream iss(out);
    std::string tok;
    while (iss >> tok) {
        char* end = nullptr;
        const char* c = tok.c_str();
        double v = std::strtod(c, &end);
        if (end && *end == '\0') { val = v; }
    }
    return val;
}

} // namespace engine
