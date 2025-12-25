#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

inline uint64_t q13_load_u64_unaligned(const unsigned char* s) {
    uint64_t result;
    memcpy(&result, s, 8);
    return result;
}

inline uint64_t q13_byte_match_mask_u64(uint64_t word, unsigned char byte_value) {
    const uint64_t ones = 0x0101010101010101ul;
    uint64_t x = word ^ (uint64_t(byte_value) * ones);
    uint64_t result = 0ul;
    for (int i = 0; i < 8; i++) {
        unsigned char byte = (x >> (i * 8)) & 0xFF;
        if (byte == 0) result |= (0x80ul << (i * 8));
    }
    return result;
}

inline int q13_effective_len_fixed_100(const unsigned char* s) {
    const int max_len = 100;
    const uint64_t ones = 0x0101010101010101ul;
    const uint64_t high = 0x8080808080808080ul;
    for (int base = 0; base < max_len; base += 8) {
        uint64_t w;
        if (base + 8 > max_len) {
            uint32_t tail = *(const uint32_t*)(s + base);
            w = (uint64_t)tail | 0xFFFFFFFF00000000ul;
        } else {
            w = q13_load_u64_unaligned(s + base);
        }
        uint64_t m = (w - ones) & (~w) & high;
        if (m == 0ul) continue;
        uint32_t lo = (uint32_t)m;
        if (lo != 0u) {
            int byte_idx = __builtin_ctz(lo) >> 3;
            int pos = base + byte_idx;
            return (pos < max_len) ? pos : max_len;
        }
        int byte_idx = (__builtin_ctz((uint32_t)(m >> 32)) >> 3) + 4;
        int pos = base + byte_idx;
        return (pos < max_len) ? pos : max_len;
    }
    return max_len;
}

inline bool q13_find_requests_after(const unsigned char* s, int start_pos, int last_requests, uint64_t REQUESTS_8) {
    if (start_pos > last_requests) return false;
    for (int base = start_pos; base <= last_requests; base += 8) {
        uint64_t w = q13_load_u64_unaligned(s + base);
        uint64_t m = q13_byte_match_mask_u64(w, (unsigned char)'r');
        uint32_t lo = (uint32_t)m;
        while (lo != 0u) {
            int pos = base + (__builtin_ctz(lo) >> 3);
            if (pos >= start_pos && pos <= last_requests) {
                if (q13_load_u64_unaligned(s + pos) == REQUESTS_8) return true;
            }
            lo &= (lo - 1u);
        }
        uint32_t hi = (uint32_t)(m >> 32);
        while (hi != 0u) {
            int pos = base + (__builtin_ctz(hi) >> 3) + 4;
            if (pos >= start_pos && pos <= last_requests) {
                if (q13_load_u64_unaligned(s + pos) == REQUESTS_8) return true;
            }
            hi &= (hi - 1u);
        }
    }
    return false;
}

inline bool q13_has_special_requests(const unsigned char* s, int comment_len) {
    const int special_len = 7, requests_len = 8;
    const int last_special = comment_len - (special_len + requests_len);
    const int last_requests = comment_len - requests_len;
    if (last_special < 0) return false;
    const uint64_t SPECIAL_MASK_7 = 0x00FFFFFFFFFFFFFFul;
    const uint64_t SPECIAL_7 = 0x006c616963657073ul; // "special" in little-endian (7 bytes)
    const uint64_t REQUESTS_8 = 0x7374736575716572ul; // "requests" in little-endian (8 bytes)
    for (int base = 0; base <= last_special; base += 8) {
        uint64_t w = q13_load_u64_unaligned(s + base);
        uint64_t m = q13_byte_match_mask_u64(w, (unsigned char)'s');
        uint32_t lo = (uint32_t)m;
        while (lo != 0u) {
            int j = base + (__builtin_ctz(lo) >> 3);
            if (j <= last_special) {
                uint64_t ws = q13_load_u64_unaligned(s + j);
                if ((ws & SPECIAL_MASK_7) == SPECIAL_7) {
                    if (q13_find_requests_after(s, j + special_len, last_requests, REQUESTS_8)) return true;
                }
            }
            lo &= (lo - 1u);
        }
        uint32_t hi = (uint32_t)(m >> 32);
        while (hi != 0u) {
            int j = base + (__builtin_ctz(hi) >> 3) + 4;
            if (j <= last_special) {
                uint64_t ws = q13_load_u64_unaligned(s + j);
                if ((ws & SPECIAL_MASK_7) == SPECIAL_7) {
                    if (q13_find_requests_after(s, j + special_len, last_requests, REQUESTS_8)) return true;
                }
            }
            hi &= (hi - 1u);
        }
    }
    return false;
}

int main() {
    std::ifstream file("GPUDBMentalBenchmark/Data/SF-1/orders.tbl");
    std::string line;
    int count_gpu_match = 0;
    while (std::getline(file, line)) {
        int col = 0;
        size_t start = 0, end;
        std::string comment;
        while ((end = line.find('|', start)) != std::string::npos) {
            if (col == 8) {
                comment = line.substr(start, end - start);
                break;
            }
            start = end + 1;
            col++;
        }
        unsigned char padded[100];
        for (int i = 0; i < 100; i++) {
            padded[i] = (i < comment.length()) ? comment[i] : '\0';
        }
        int effective_len = q13_effective_len_fixed_100(padded);
        if (q13_has_special_requests(padded, effective_len)) count_gpu_match++;
    }
    std::cout << "GPU logic matches: " << count_gpu_match << "\n";
    std::cout << "Expected (grep): 16082\n";
    std::cout << "Difference: " << (count_gpu_match - 16082) << "\n";
    return 0;
}
