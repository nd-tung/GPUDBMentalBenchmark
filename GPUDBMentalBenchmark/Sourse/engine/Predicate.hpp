// Lightweight predicate pre-parsing and evaluation
#pragma once
#include <string>
#include <vector>
#include <functional>

namespace engine::expr {

using RowGetter = std::function<double(std::size_t,const std::string&)>;
using IntGetter = std::function<long long(std::size_t,const std::string&)>;
using ExistsFn = std::function<bool(const std::string&)>;

enum class CompOp { LT, LE, GT, GE, EQ };

struct Clause {
    std::string ident;      // left-hand side column identifier
    CompOp op;              // comparison operator
    bool isDate = false;    // true if RHS was a DATE literal
    double num = 0.0;       // numeric literal (if !isDate)
    long long date = 0;     // date literal encoded as YYYYMMDD (if isDate)
};

// Parse a conjunction of simple comparisons separated by AND.
// Supports: <, <=, >, >=, = with numeric literals or DATE 'YYYY-MM-DD'
// Returns vector of Clause; if a comparison cannot be parsed it is skipped.
std::vector<Clause> parse_predicate(const std::string& predicate, const ExistsFn& exists);

// Evaluate already parsed clauses for a given row using provided accessors.
bool eval_predicate(const std::vector<Clause>& clauses,
                    std::size_t rowIndex,
                    const RowGetter& getFloatLike,
                    const IntGetter& getIntLike);

} // namespace engine::expr
