#include "ExprEval.hpp"
#include <cctype>
#include <stack>
#include <regex>
#include <set>

namespace engine::expr {

static bool is_ident_char(char c){ return std::isalnum(static_cast<unsigned char>(c)) || c=='_' || c=='.'; }

std::vector<Token> tokenize_arith(const std::string& expr) {
    std::vector<Token> out; out.reserve(expr.size());
    std::size_t i=0, n=expr.size();
    while (i<n) {
        char c = expr[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (std::isdigit(static_cast<unsigned char>(c)) || (c=='.')) {
            std::size_t j=i+1;
            while (j<n && (std::isdigit(static_cast<unsigned char>(expr[j])) || expr[j]=='.' || expr[j]=='e' || expr[j]=='E' || expr[j]=='+' || expr[j]=='-')) {
                if ((expr[j]=='+'||expr[j]=='-') && (expr[j-1]!='e' && expr[j-1]!='E')) break;
                ++j;
            }
            out.push_back({Token::Type::Number, expr.substr(i,j-i)}); i=j; continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c=='_') {
            std::size_t j=i+1; while (j<n && is_ident_char(expr[j])) ++j;
            out.push_back({Token::Type::Ident, expr.substr(i,j-i)}); i=j; continue;
        }
        if (c=='(') { out.push_back({Token::Type::LParen, "("}); ++i; continue; }
        if (c==')') { out.push_back({Token::Type::RParen, ")"}); ++i; continue; }
        if (c=='+'||c=='-'||c=='*'||c=='/') { out.push_back({Token::Type::Op, std::string(1,c)}); ++i; continue; }
        ++i;
    }
    return out;
}

static int precedence(const std::string& op) {
    if (op=="+"||op=="-") return 1; if (op=="*"||op=="/") return 2; return 0;
}

std::vector<Token> to_rpn(const std::vector<Token>& tokens) {
    std::vector<Token> out; out.reserve(tokens.size());
    std::vector<Token> st;
    for (const auto& t: tokens) {
        if (t.type==Token::Type::Number || t.type==Token::Type::Ident) { out.push_back(t); }
        else if (t.type==Token::Type::Op) {
            while (!st.empty() && st.back().type==Token::Type::Op && precedence(st.back().text)>=precedence(t.text)) { out.push_back(st.back()); st.pop_back(); }
            st.push_back(t);
        } else if (t.type==Token::Type::LParen) { st.push_back(t); }
        else if (t.type==Token::Type::RParen) {
            while (!st.empty() && st.back().type!=Token::Type::LParen) { out.push_back(st.back()); st.pop_back(); }
            if (!st.empty() && st.back().type==Token::Type::LParen) st.pop_back();
        }
    }
    while (!st.empty()) { if (st.back().type!=Token::Type::LParen) out.push_back(st.back()); st.pop_back(); }
    return out;
}

double eval_rpn(const std::vector<Token>& rpn, std::size_t rowIndex, const RowGetter& getVal) {
    std::vector<double> st; st.reserve(rpn.size());
    for (const auto& t: rpn) {
        if (t.type==Token::Type::Number) {
            st.push_back(std::stod(t.text));
        } else if (t.type==Token::Type::Ident) {
            st.push_back(getVal(rowIndex, t.text));
        } else if (t.type==Token::Type::Op) {
            if (st.size()<2) return 0.0; double b=st.back(); st.pop_back(); double a=st.back(); st.pop_back();
            if (t.text=="+") st.push_back(a+b); else if (t.text=="-") st.push_back(a-b); else if (t.text=="*") st.push_back(a*b); else if (t.text=="/") st.push_back(b!=0.0? a/b : 0.0);
        }
    }
    return st.empty()? 0.0 : st.back();
}

std::set<std::string> collect_idents(const std::string& s) {
    std::set<std::string> ids; std::size_t i=0,n=s.size();
    while (i<n) {
        char c=s[i]; if (std::isalpha(static_cast<unsigned char>(c))||c=='_') { std::size_t j=i+1; while (j<n && is_ident_char(s[j])) ++j; ids.insert(s.substr(i,j-i)); i=j; }
        else ++i;
    }
    return ids;
}

int parse_date_yyyymmdd(const std::string& s) {
    std::string t; t.reserve(8);
    for(char c: s) if (std::isdigit(static_cast<unsigned char>(c))) t.push_back(c);
    if (t.size()==8) return std::stoi(t);
    return 0;
}

static bool parse_comparison(const std::string& clause, std::string& ident, std::string& op, std::string& rhsRaw, bool& isDate) {
    std::regex re("^\\s*([A-Za-z_][A-Za-z0-9_\.]*)\\s*(<=|>=|=|<|>)\\s*(DATE\\s*'[^']+'|[+-]?[0-9]*\\.?[0-9]+)\\s*$", std::regex::icase);
    std::smatch m; if (!std::regex_match(clause, m, re)) return false;
    ident = m[1].str(); op = m[2].str(); rhsRaw = m[3].str();
    std::string low = rhsRaw; for(char& c: low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    isDate = low.rfind("date",0)==0;
    return true;
}

bool eval_predicate_conjunction(const std::string& predicate,
                               std::size_t rowIndex,
                               const RowGetter& getFloatLike,
                               const IntGetter& getIntLike,
                               const ExistsFn& exists) {
    if (predicate.empty()) return true;
    std::vector<std::string> clauses; clauses.reserve(8);
    {
        std::regex delim("\\s+and\\s+", std::regex::icase);
        std::sregex_token_iterator it(predicate.begin(), predicate.end(), delim, -1);
        std::sregex_token_iterator end;
        for (; it != end; ++it) {
            std::string s = it->str();
            // trim
            auto l = s.find_first_not_of(" \t\n\r");
            auto r = s.find_last_not_of(" \t\n\r");
            if (l==std::string::npos) continue; // empty
            clauses.push_back(s.substr(l, r-l+1));
        }
    }
    for (auto& c : clauses) {
        std::string ident, op, rhs; bool isDate=false;
        if (!parse_comparison(c, ident, op, rhs, isDate)) return false;
        if (!exists(ident)) return false;
        if (isDate) {
            auto q1 = rhs.find("'"); auto q2 = rhs.rfind("'");
            std::string lit = (q1!=std::string::npos && q2!=std::string::npos && q2>q1)? rhs.substr(q1+1, q2-q1-1) : rhs;
            long long r = parse_date_yyyymmdd(lit);
            long long l = getIntLike(rowIndex, ident);
            if (op=="<") { if (!(l < r)) return false; }
            else if (op=="<=") { if (!(l <= r)) return false; }
            else if (op==">") { if (!(l > r)) return false; }
            else if (op==">=") { if (!(l >= r)) return false; }
            else if (op=="=") { if (!(l == r)) return false; }
            else return false;
        } else {
            double r = std::stod(rhs);
            double l = getFloatLike(rowIndex, ident);
            if (op=="<") { if (!(l < r)) return false; }
            else if (op=="<=") { if (!(l <= r)) return false; }
            else if (op==">") { if (!(l > r)) return false; }
            else if (op==">=") { if (!(l >= r)) return false; }
            else if (op=="=") { if (!(l == r)) return false; }
            else return false;
        }
    }
    return true;
}

} // namespace engine::expr
