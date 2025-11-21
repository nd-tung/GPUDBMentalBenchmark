#pragma once
#include <string>
#include <vector>
#include <set>
#include <functional>

namespace engine::expr {

struct Token {
    enum class Type { Number, Ident, Op, LParen, RParen } type;
    std::string text;
};

using RowGetter = std::function<double(std::size_t, const std::string&)>;

std::vector<Token> tokenize_arith(const std::string& expr);
std::vector<Token> to_rpn(const std::vector<Token>& tokens);
double eval_rpn(const std::vector<Token>& rpn, std::size_t rowIndex, const RowGetter& getVal);

std::set<std::string> collect_idents(const std::string& s);

// Very small predicate evaluator supporting conjunctions of simple comparisons connected by AND.
// Supported clause forms:
//   <ident> <op> <number>
//   <ident> <op> DATE 'YYYY-MM-DD'
// where <op> is one of: <, <=, >, >=, =
// Returns true if row satisfies predicate. Empty predicate => true.
using IntGetter = std::function<long long(std::size_t, const std::string&)>;
using ExistsFn = std::function<bool(const std::string&)>;
bool eval_predicate_conjunction(const std::string& predicate,
                               std::size_t rowIndex,
                               const RowGetter& getFloatLike,
                               const IntGetter& getIntLike,
                               const ExistsFn& exists);

int parse_date_yyyymmdd(const std::string& s);

} // namespace engine::expr
