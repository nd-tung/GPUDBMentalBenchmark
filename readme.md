# Benchmark Results Summary

## Table 1: Selection Performance vs. Scale Factor

| Scale Factor | Data Size   | GPU Time (ms) | Bandwidth (GB/s) |
|--------------|-------------|---------------|------------------|
| SF-1         | 22.89 MB    | 0.93          | 24.15            |
| SF-10        | 228.82 MB   | 8.15          | 27.43            |

## Table 2: Aggregation Performance (Optimized Kernel)

| Scale Factor | Data Size   | GPU Time (ms) | Bandwidth (GB/s) |
|--------------|-------------|---------------|------------------|
| SF-1         | 22.89 MB    | 1.32          | 16.89            |
| SF-10        | 228.82 MB   | 4.20          | 53.21            |

## Table 3: Hash Join Performance vs. Scale Factor

| Scale Factor | Build Time (ms) | Probe Time (ms) | Total GPU Time (ms) |
|--------------|------------------|-----------------|---------------------|
| SF-1         | 2.97             | 10.94           | 13.91               |
| SF-10        | 20.05            | 40.30           | 60.35               |

## Table 4: TPC-H Query 1 Performance (Optimized Pipeline)

| Scale Factor | Total Rows Processed | GPU Time (ms) |
|--------------|-----------------------|---------------|
| SF-1         | ~6 Million           | 46.7          |
| SF-10        | ~60 Million          | 119.9         |

