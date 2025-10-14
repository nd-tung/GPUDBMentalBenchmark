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
| TPC-H Query 1 | 48.74 ms | 93.45 ms | **2x** |

### SF-10 Dataset (60M rows)
| Operation | GPU (Metal) | DuckDB | Speedup |
|-----------|-------------|---------|---------|
| Selection < 1000 | 8.15 ms | 210.79 ms | **26x** |
| Selection < 10000 | 8.05 ms | 95.50 ms | **12x** |
| Selection < 50000 | 8.11 ms | 95.02 ms | **12x** |
| SUM Aggregation | 4.66 ms | 120.01 ms | **26x** |
| Hash Join | 58.12 ms | 486.42 ms | **8x** |
| TPC-H Query 1 | 115.34 ms | 576.44 ms | **5x** |

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
./performance_comparison.sh
```



