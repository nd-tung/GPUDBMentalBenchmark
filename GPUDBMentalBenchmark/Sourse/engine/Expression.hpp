#pragma once
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <cctype>
#include <stdexcept>
#include <map>
#include <functional>

namespace engine::expr {

// Expression tree nodes
struct ExprNode {
    enum class Type { Column, Literal, BinaryOp } type;
    
    // Column reference
    std::string column;
    
    // Literal value
    float literal = 0.0f;
    
    // Binary operation
    char op = 0;  // '+', '-', '*', '/'
    std::shared_ptr<ExprNode> left;
    std::shared_ptr<ExprNode> right;
    
    explicit ExprNode(const std::string& col) : type(Type::Column), column(col) {}
    explicit ExprNode(float val) : type(Type::Literal), literal(val) {}
    ExprNode(char o, std::shared_ptr<ExprNode> l, std::shared_ptr<ExprNode> r) 
        : type(Type::BinaryOp), op(o), left(l), right(r) {}
};

// Token for RPN (Reverse Polish Notation) representation - matches Metal struct
struct ExprToken {
    uint32_t type;       // 0:column_ref 1:literal 2:operator
    uint32_t colIndex;   // if type==0, column index
    float literal;       // if type==1, literal value
    uint32_t op;         // if type==2, operator: 0:+ 1:- 2:* 3:/
};

// Simple tokenizer for expressions
class ExprTokenizer {
    std::string text;
    size_t pos = 0;
    
public:
    explicit ExprTokenizer(const std::string& s) : text(s) {}
    
    struct Token {
        enum class Type { Number, Ident, Op, LParen, RParen, End } type;
        std::string value;
        char op;
        float num;
    };
    
    void skipWhitespace() {
        while (pos < text.size() && std::isspace(text[pos])) ++pos;
    }
    
    Token next() {
        skipWhitespace();
        if (pos >= text.size()) return {Token::Type::End, "", 0, 0.0f};
        
        char c = text[pos];
        
        // Operators
        if (c == '+' || c == '-' || c == '*' || c == '/') {
            ++pos;
            return {Token::Type::Op, std::string(1, c), c, 0.0f};
        }
        
        // Parentheses
        if (c == '(') { ++pos; return {Token::Type::LParen, "(", 0, 0.0f}; }
        if (c == ')') { ++pos; return {Token::Type::RParen, ")", 0, 0.0f}; }
        
        // Numbers
        if (std::isdigit(c) || c == '.') {
            size_t start = pos;
            while (pos < text.size() && (std::isdigit(text[pos]) || text[pos] == '.')) ++pos;
            std::string num_str = text.substr(start, pos - start);
            return {Token::Type::Number, num_str, 0, std::stof(num_str)};
        }
        
        // Identifiers (column names)
        if (std::isalpha(c) || c == '_') {
            size_t start = pos;
            while (pos < text.size() && (std::isalnum(text[pos]) || text[pos] == '_')) ++pos;
            std::string ident = text.substr(start, pos - start);
            return {Token::Type::Ident, ident, 0, 0.0f};
        }
        
        throw std::runtime_error("Unexpected character in expression: " + std::string(1, c));
    }
};

// Recursive descent parser for arithmetic expressions
class ExprParser {
    ExprTokenizer tokenizer;
    ExprTokenizer::Token current;
    
    void advance() { current = tokenizer.next(); }
    
    std::shared_ptr<ExprNode> parsePrimary() {
        if (current.type == ExprTokenizer::Token::Type::Number) {
            auto node = std::make_shared<ExprNode>(current.num);
            advance();
            return node;
        }
        if (current.type == ExprTokenizer::Token::Type::Ident) {
            auto node = std::make_shared<ExprNode>(current.value);
            advance();
            return node;
        }
        if (current.type == ExprTokenizer::Token::Type::LParen) {
            advance();  // skip '('
            auto node = parseExpression();
            if (current.type != ExprTokenizer::Token::Type::RParen) {
                throw std::runtime_error("Expected ')' in expression");
            }
            advance();  // skip ')'
            return node;
        }
        throw std::runtime_error("Expected number, identifier, or '(' in expression");
    }
    
    std::shared_ptr<ExprNode> parseTerm() {
        auto left = parsePrimary();
        
        while (current.type == ExprTokenizer::Token::Type::Op &&
               (current.op == '*' || current.op == '/')) {
            char op = current.op;
            advance();
            auto right = parsePrimary();
            left = std::make_shared<ExprNode>(op, left, right);
        }
        
        return left;
    }
    
    std::shared_ptr<ExprNode> parseExpression() {
        auto left = parseTerm();
        
        while (current.type == ExprTokenizer::Token::Type::Op &&
               (current.op == '+' || current.op == '-')) {
            char op = current.op;
            advance();
            auto right = parseTerm();
            left = std::make_shared<ExprNode>(op, left, right);
        }
        
        return left;
    }
    
public:
    explicit ExprParser(const std::string& expr) : tokenizer(expr) {
        advance();
    }
    
    std::shared_ptr<ExprNode> parse() {
        return parseExpression();
    }
};

// Convert expression tree to RPN (Reverse Polish Notation) for GPU evaluation
// Takes column index map to resolve column names immediately
inline void exprToRPN(const std::shared_ptr<ExprNode>& node, 
                      std::vector<ExprToken>& rpn,
                      const std::map<std::string, uint32_t>& colMap) {
    if (!node) return;
    
    if (node->type == ExprNode::Type::Column) {
        // Column reference - resolve index immediately
        uint32_t colIndex = 0;
        auto it = colMap.find(node->column);
        if (it != colMap.end()) {
            colIndex = it->second;
        }
        rpn.push_back({0, colIndex, 0.0f, 0});  // type=0 (column_ref)
    } else if (node->type == ExprNode::Type::Literal) {
        // Literal value
        rpn.push_back({1, 0, node->literal, 0});  // type=1 (literal)
    } else if (node->type == ExprNode::Type::BinaryOp) {
        // Post-order traversal for RPN
        exprToRPN(node->left, rpn, colMap);
        exprToRPN(node->right, rpn, colMap);
        
        // Operator
        uint32_t op_code = 0;
        switch (node->op) {
            case '+': op_code = 0; break;
            case '-': op_code = 1; break;
            case '*': op_code = 2; break;
            case '/': op_code = 3; break;
        }
        rpn.push_back({2, 0, 0.0f, op_code});  // type=2 (operator)
    }
}

// Extract all column names from expression tree
inline void extractColumns(const std::shared_ptr<ExprNode>& node, std::vector<std::string>& columns) {
    if (!node) return;
    
    if (node->type == ExprNode::Type::Column) {
        // Check if column already in list
        bool found = false;
        for (const auto& col : columns) {
            if (col == node->column) { found = true; break; }
        }
        if (!found) columns.push_back(node->column);
    } else if (node->type == ExprNode::Type::BinaryOp) {
        extractColumns(node->left, columns);
        extractColumns(node->right, columns);
    }
}

// Build column index map from column order
inline std::map<std::string, uint32_t> buildColumnMap(const std::vector<std::string>& columnOrder) {
    std::map<std::string, uint32_t> colMap;
    for (size_t i = 0; i < columnOrder.size(); ++i) {
        colMap[columnOrder[i]] = static_cast<uint32_t>(i);
    }
    return colMap;
}

// Helper to parse expression string and return both tree and RPN with columns resolved
struct ParsedExpression {
    std::shared_ptr<ExprNode> tree;
    std::vector<ExprToken> rpn;
    std::vector<std::string> columns;  // All referenced columns in order
    
    static ParsedExpression parse(const std::string& exprStr) {
        ParsedExpression result;
        
        // Parse to tree
        ExprParser parser(exprStr);
        result.tree = parser.parse();
        
        // Extract columns
        extractColumns(result.tree, result.columns);
        
        // Build column map
        auto colMap = buildColumnMap(result.columns);
        
        // Convert to RPN with column indices resolved
        exprToRPN(result.tree, result.rpn, colMap);
        
        return result;
    }
};

} // namespace engine::expr
