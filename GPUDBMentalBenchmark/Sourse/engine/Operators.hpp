#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <functional>

namespace engine {

struct BufferView {
    const void* data;
    std::size_t count;
    std::size_t stride;
};

struct MutableBufferView {
    void* data;
    std::size_t count;
    std::size_t stride;
};

struct KernelConfig {
    std::string functionName;
    std::size_t threads;
    std::size_t threadgroups;
};

class FilterProject {
public:
    struct Params {
        uint32_t eqValue = 0;
    };

    void init(const KernelConfig& cfg);
    void dispatch(const BufferView& in, MutableBufferView maskOut, const Params& p);

private:
    KernelConfig cfg_{};
};

class HashJoinU32 {
public:
    struct BuildInput { BufferView keys; BufferView payloads; };
    struct ProbeInput { BufferView keys; };
    struct Output { MutableBufferView payloads; };

    void init(const KernelConfig& buildCfg, const KernelConfig& probeCfg, std::size_t capacity);
    void build(const BuildInput& in);
    void probe(const ProbeInput& in, Output out);

private:
    KernelConfig buildCfg_{};
    KernelConfig probeCfg_{};
    std::size_t capacity_{};
};

class GroupBySumF32 {
public:
    struct Input { BufferView keys; BufferView vals; };
    struct Output { MutableBufferView bucketKeys; MutableBufferView bucketCounts; MutableBufferView bucketSumsBits; };

    void init(const KernelConfig& cfg, std::size_t bucketsPow2);
    void aggregate(const Input& in, Output out);

private:
    KernelConfig cfg_{};
    std::size_t bucketMask_{};
};

class FilterQ6 {
public:
    struct Params {
        int start_date;   // inclusive
        int end_date;     // exclusive
        float min_discount;
        float max_discount;
        float max_quantity;
    };

    void init(const KernelConfig& cfg) { cfg_ = cfg; }

    double computeRevenue(const BufferView& shipdate,
                          const BufferView& discount,
                          const BufferView& quantity,
                          const BufferView& extendedprice,
                          const Params& p);

private:
    KernelConfig cfg_{};
};

} // namespace engine
