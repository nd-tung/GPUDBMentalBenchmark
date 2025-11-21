#include "Predicate.hpp"
#include "ExprEval.hpp" // for parse_date_yyyymmdd
#include <regex>
#include <cctype>
#include <iostream>

namespace engine::expr {

static bool parse_single(const std::string& clause, Clause& out, const ExistsFn& exists) {
    std::regex re("^\\s*([A-Za-z_][A-Za-z0-9_\\.]*)\\s*(<=|>=|=|<|>)\\s*(DATE\\s*'[^']+'|[+-]?[0-9]*\\.?[0-9]+)\\s*$", std::regex::icase);
    std::smatch m; if (!std::regex_match(clause, m, re)) return false;
    std::string ident = m[1].str(); if (!exists(ident)) return false;
    std::string op = m[2].str(); std::string rhs = m[3].str();
    Clause c; c.ident = ident;
    if (op=="<") c.op = CompOp::LT; else if (op=="<=") c.op = CompOp::LE; else if (op==">") c.op = CompOp::GT; else if (op==">=") c.op = CompOp::GE; else c.op = CompOp::EQ;
    std::string low = rhs; for(char& ch: low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (low.rfind("date",0)==0) {
        auto q1 = rhs.find("'"); auto q2 = rhs.rfind("'");
        std::string lit = (q1!=std::string::npos && q2!=std::string::npos && q2>q1)? rhs.substr(q1+1, q2-q1-1) : rhs;
        c.isDate = true; c.date = parse_date_yyyymmdd(lit);
    } else {
        c.isDate = false; c.num = std::stod(rhs);
    }
    out = c; return true;
}

std::vector<Clause> parse_predicate(const std::string& predicate, const ExistsFn& exists) {
    std::vector<Clause> out; if (predicate.empty()) return out;
    
    // Split by both AND and OR, keeping track of which delimiter was used
    std::regex delim("\\s+(and|or)\\s+", std::regex::icase);
    std::sregex_token_iterator it(predicate.begin(), predicate.end(), delim, {-1, 1}), end;
    
    std::vector<std::string> tokens;
    for (; it != end; ++it) {
        if (!it->str().empty()) tokens.push_back(it->str());
    }
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string s = tokens[i];
        auto l = s.find_first_not_of(" \t\n\r");
        auto r = s.find_last_not_of(" \t\n\r");
        if (l == std::string::npos) continue;
        s = s.substr(l, r - l + 1);
        
        // Check if this is a delimiter (AND/OR)
        std::string lower = s;
        for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (lower == "and" || lower == "or") {
            // Mark the previous clause as OR'd with the next if this is OR
            if (lower == "or" && !out.empty()) {
                out.back().isOrNext = true;
            }
        } else {
            // This is a comparison clause
            Clause c;
            if (parse_single(s, c, exists)) {
                out.push_back(c);
            }
        }
    }
    return out;
}

static bool cmp_num(double l, CompOp op, double r) {
    // Cast to float32 for consistent precision with GPU evaluation
    float lf = static_cast<float>(l);
    float rf = static_cast<float>(r);
    switch(op){case CompOp::LT: return lf<rf; case CompOp::LE: return lf<=rf; case CompOp::GT: return lf>rf; case CompOp::GE: return lf>=rf; case CompOp::EQ: return lf==rf;} return false;
}
static bool cmp_int(long long l, CompOp op, long long r) {
    switch(op){case CompOp::LT: return l<r; case CompOp::LE: return l<=r; case CompOp::GT: return l>r; case CompOp::GE: return l>=r; case CompOp::EQ: return l==r;} return false;
}

bool eval_predicate(const std::vector<Clause>& clauses,
                    std::size_t rowIndex,
                    const RowGetter& getFloatLike,
                    const IntGetter& getIntLike) {
    if (clauses.empty()) return true;
    
    // Evaluate clauses with OR/AND logic
    // Group consecutive clauses connected by OR, evaluate groups with AND
    bool groupResult = true;
    bool finalResult = true;
    
    for (size_t i = 0; i < clauses.size(); ++i) {
        const auto& c = clauses[i];
        bool clauseResult;
        
        if (c.isDate) {
            long long l = getIntLike(rowIndex, c.ident);
            clauseResult = cmp_int(l, c.op, c.date);
        } else {
            double l = getFloatLike(rowIndex, c.ident);
            clauseResult = cmp_num(l, c.op, c.num);
        }
        
        if (i == 0) {
            // First clause
            groupResult = clauseResult;
        } else if (clauses[i-1].isOrNext) {
            // Previous clause was OR'd with this one
            groupResult = groupResult || clauseResult;
        } else {
            // Previous clause was AND'd with this one - finish previous group
            finalResult = finalResult && groupResult;
            if (!finalResult) return false; // Short circuit
            groupResult = clauseResult;
        }
    }
    
    // Don't forget the last group
    finalResult = finalResult && groupResult;
    return finalResult;
}

} // namespace engine::expr
