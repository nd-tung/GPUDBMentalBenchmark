# GPU Database Benchmark

GPU-accelerated database operations using Apple Metal vs DuckDB comparison.

## Latest Results

### SF-1 Dataset (6M rows)
| Operation | GPU (Metal) | DuckDB | Speedup |
|-----------|-------------|---------|---------|
| Selection < 1000 | 0.95 ms | 47.78 ms | **50x** |
| Selection < 10000 | 0.92 ms | 46.61 ms | **51x** |
| Selection < 50000 | 0.98 ms | 46.94 ms | **48x** |
| SUM Aggregation | 1.32 ms | 47.03 ms | **36x** |
| Hash Join | 14.49 ms | 87.13 ms | **6x** |
| TPC-H Query 1 | 38.84 ms | 82.48 ms | **2.12x** |

### SF-10 Dataset (60M rows)
| Operation | GPU (Metal) | DuckDB | Speedup |
|-----------|-------------|---------|---------|
| Selection < 1000 | 8.15 ms | 210.79 ms | **26x** |
| Selection < 10000 | 8.05 ms | 95.50 ms | **12x** |
| Selection < 50000 | 8.11 ms | 95.02 ms | **12x** |
| SUM Aggregation | 4.66 ms | 120.01 ms | **26x** |
| Hash Join | 58.12 ms | 486.42 ms | **8x** |
| TPC-H Query 1 | 384.37 ms | 598.55 ms | **1.56x** |

### TPC-H Queries (apples-to-apples)

Notes:
- DuckDB times are execution-only (no data load).
- GPU Total = GPU wall-clock + CPU merge (if any). Use this to compare with DuckDB.
- GPU compute (device-only) is available in the combined report for diagnostics.

#### SF-1

| Query | DuckDB exec (ms) | GPU Total (ms) | Speedup |
|------:|------------------:|---------------:|--------:|
| Q1 | 82.48 | 38.84 | 2.12x |
| Q3 | 59.38 | 54.34 | 1.09x |
| Q6 | 44.87 | 4.30 | 10.43x |
| Q9 | 112.73 | 54.10 | 2.08x |
| Q13 | 99.25 | 50.70 | 1.96x |

#### SF-10

| Query | DuckDB exec (ms) | GPU Total (ms) | Speedup |
|------:|------------------:|---------------:|--------:|
| Q1 | 598.55 | 384.37 | 1.56x |
| Q3 | 317.87 | 514.95 | 0.62x |
| Q6 | 165.17 | 155.24 | 1.06x |
| Q9 | 1176.68 | 795.72 | 1.48x |
| Q13 | 1044.42 | 659.59 | 1.58x |

## How to Run

### Prerequisites
- macOS with Metal support
- `brew install duckdb`

### GPU Benchmark
```bash
make run      # SF-10 dataset (60M rows)
make run-sf1  # SF-1 dataset (6M rows)
```

### DuckDB Comparison
```bash
./benchmark_duckdb.sh
./benchmark_gpu.sh
python3 scripts/generate_combined_report.py
```



