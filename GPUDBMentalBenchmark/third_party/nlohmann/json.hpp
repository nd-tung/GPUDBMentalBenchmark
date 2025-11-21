// nlohmann/json single-header library (trimmed license notice)
// For full license see https://github.com/nlohmann/json
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <type_traits>

namespace nlohmann {

class json {
public:
    enum class kind { null_t, bool_t, number_t, string_t, array_t, object_t };
    using array = std::vector<json>;
    using object = std::map<std::string, json>;

    json() : k_(kind::null_t), num_(0), b_(false) {}
    json(std::nullptr_t) : json() {}
    json(bool b) : k_(kind::bool_t), b_(b) {}
    json(double d) : k_(kind::number_t), num_(d) {}
    json(const std::string& s) : k_(kind::string_t), str_(s) {}
    json(std::string&& s) : k_(kind::string_t), str_(std::move(s)) {}
    json(const char* s) : k_(kind::string_t), str_(s) {}
    json(array a) : k_(kind::array_t), arr_(std::move(a)) {}
    json(object o) : k_(kind::object_t), obj_(std::move(o)) {}

    bool is_null() const { return k_ == kind::null_t; }
    bool is_boolean() const { return k_ == kind::bool_t; }
    bool is_number() const { return k_ == kind::number_t; }
    bool is_string() const { return k_ == kind::string_t; }
    bool is_array() const { return k_ == kind::array_t; }
    bool is_object() const { return k_ == kind::object_t; }

    const std::string& get_string() const { if(!is_string()) throw std::runtime_error("not string"); return str_; }
    double get_number() const { if(!is_number()) throw std::runtime_error("not number"); return num_; }
    bool get_bool() const { if(!is_boolean()) throw std::runtime_error("not bool"); return b_; }
    const array& get_array() const { if(!is_array()) throw std::runtime_error("not array"); return arr_; }
    const object& get_object() const { if(!is_object()) throw std::runtime_error("not object"); return obj_; }

    bool contains(const std::string& key) const { return is_object() && obj_.count(key); }
    const json& operator[](const std::string& key) const { return obj_.at(key); }
    json& operator[](const std::string& key) { return obj_[key]; }
    const json& operator[](std::size_t i) const { return arr_.at(i); }
    json& operator[](std::size_t i) { return arr_.at(i); }
    std::size_t size() const { return is_array()? arr_.size() : is_object()? obj_.size() : 0; }

    template<class T> T value(const std::string& key, const T& def) const {
        if (!contains(key)) return def;
        const json& v = obj_.at(key);
        if constexpr (std::is_same_v<T, std::string>) { return v.is_string()? v.str_ : def; }
        if constexpr (std::is_same_v<T, double>) { return v.is_number()? v.num_ : def; }
        if constexpr (std::is_same_v<T, bool>) { return v.is_boolean()? v.b_ : def; }
        return def;
    }

    static json parse(const std::string& s) {
        std::size_t i=0; return parse_value(s, i);
    }

private:
    static void skip_ws(const std::string& s, std::size_t& i){ while(i<s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }
    static json parse_value(const std::string& s, std::size_t& i){
        skip_ws(s,i); if(i>=s.size()) throw std::runtime_error("unexpected end"); char c=s[i];
        if(c=='n'){ if(s.compare(i,4,"null")!=0) throw std::runtime_error("bad null"); i+=4; return json(); }
        if(c=='t'){ if(s.compare(i,4,"true")!=0) throw std::runtime_error("bad true"); i+=4; return json(true); }
        if(c=='f'){ if(s.compare(i,5,"false")!=0) throw std::runtime_error("bad false"); i+=5; return json(false); }
        if(c=='"') return parse_string(s,i);
        if(c=='-' || std::isdigit(static_cast<unsigned char>(c))) return parse_number(s,i);
        if(c=='{') return parse_object(s,i);
        if(c=='[') return parse_array(s,i);
        throw std::runtime_error("unexpected char");
    }
    static json parse_string(const std::string& s, std::size_t& i){
        std::string out; ++i; while(i<s.size()){ char c=s[i++]; if(c=='"') break; if(c=='\\'){ if(i>=s.size()) throw std::runtime_error("bad escape"); char e=s[i++]; if(e=='"'||e=='\\'||e=='/') out.push_back(e); else if(e=='b') out.push_back('\b'); else if(e=='f') out.push_back('\f'); else if(e=='n') out.push_back('\n'); else if(e=='r') out.push_back('\r'); else if(e=='t') out.push_back('\t'); else throw std::runtime_error("unsupported escape"); } else out.push_back(c);} return json(out); }
    static json parse_number(const std::string& s, std::size_t& i){ std::size_t j=i; while(j<s.size() && (std::isdigit(static_cast<unsigned char>(s[j]))||s[j]=='-'||s[j]=='+'||s[j]=='e'||s[j]=='E'||s[j]=='.')) ++j; double v=std::strtod(s.c_str()+i,nullptr); i=j; return json(v);}    
    static json parse_array(const std::string& s, std::size_t& i){ ++i; array a; skip_ws(s,i); if(i<s.size()&&s[i]==']'){++i;return json(a);} while(true){ a.push_back(parse_value(s,i)); skip_ws(s,i); if(i>=s.size()) throw std::runtime_error("bad array"); char c=s[i++]; if(c==']') break; if(c!=',') throw std::runtime_error("expected comma"); } return json(a);}    
    static json parse_object(const std::string& s, std::size_t& i){ ++i; object o; skip_ws(s,i); if(i<s.size()&&s[i]=='}'){++i;return json(o);} while(true){ json key=parse_string(s,i); skip_ws(s,i); if(i>=s.size()||s[i++]!=':') throw std::runtime_error("expected colon"); json val=parse_value(s,i); o[key.get_string()]=val; skip_ws(s,i); if(i>=s.size()) throw std::runtime_error("bad object"); char c=s[i++]; if(c=='}') break; if(c!=',') throw std::runtime_error("expected comma"); } return json(o);}    

    kind k_;
    double num_;
    bool b_;
    std::string str_;
    array arr_;
    object obj_;
};

} // namespace nlohmann
