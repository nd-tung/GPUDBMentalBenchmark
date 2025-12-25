# GPU Database Benchmark

GPU-accelerated database operations using Apple Metal vs DuckDB vs CedarDB comparison.

## Latest Benchmark Results

**Timestamp**: 2025-12-25  
**Methodology**: Warm cache (data pre-loaded), execution time only

### GPU Metal Results (Timestamp: 20251225_115212)

#### SF-1 Dataset (6M lineitem rows)
| Query | GPU Time (ms) | Wall Clock (ms) | CPU Merge (ms) | Execution Time (GPU+CPU) (ms) |
|-------|--------------|----------------|----------------|-------------------------------|
| Q1    | 34.00        | 37.82          | 0.00           | 34.00                         |
| Q3    | 9.92         | 24.51          | 1.21           | 11.13                         |
| Q6    | 1.72         | 4.82           | 0.00           | 1.72                          |
| Q9    | 29.91        | 38.21          | 0.02           | 29.93                         |
| Q13   | 31.21        | 37.38          | 1.75           | 32.96                         |

#### SF-10 Dataset (60M lineitem rows)
| Query | GPU Time (ms) | Wall Clock (ms) | CPU Merge (ms) | Execution Time (GPU+CPU) (ms) |
|-------|--------------|----------------|----------------|-------------------------------|
| Q1    | 166.57       | 363.28         | 0.00           | 166.57                        |
| Q3    | 48.39        | 369.24         | 19.11          | 67.50                         |
| Q6    | 12.91        | 91.78          | 0.00           | 12.91                         |
| Q9    | 389.64       | 803.61         | 0.21           | 389.85                        |
| Q13   | 225.58       | 292.40         | 37.44          | 263.02                        |

### DuckDB Results (Timestamp: 20251225_115212)

#### SF-1 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 50                  |
| Q3    | 26                  |
| Q6    | 12                  |
| Q9    | 71                  |
| Q13   | 67                  |

#### SF-10 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 554                 |
| Q3    | 256                 |
| Q6    | 98                  |
| Q9    | 762                 |
| Q13   | 701                 |

### CedarDB Results (Timestamps: 20251103_164713 for SF-1, 20251103_153654 for SF-10)

#### SF-1 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 46                  |
| Q3    | 28                  |
| Q6    | 2                   |
| Q9    | 113                 |
| Q13   | 121                 |

#### SF-10 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 229                 |
| Q3    | 399                 |
| Q6    | 23                  |
| Q9    | 1114                |
| Q13   | 1687                |

## How to Run Benchmarks

### Prerequisites
```bash
# Install DuckDB
brew install duckdb

# Install CedarDB via Docker
docker pull cedardb/cedardb
```

### Quick Start
```bash
# 1. Build GPU benchmark
make

# 2. Generate test data (if not already generated)
./create_tpch_data.sh

# 3. Run all benchmarks
./benchmark_gpu.sh SF-1 SF-10
./benchmark_duckdb.sh SF-1 SF-10
./benchmark_cedardb.sh SF-1 SF-10  # Requires Docker

# 4. View results
cat benchmark_results/gpu_results.csv
cat benchmark_results/duckdb_results.csv
cat benchmark_results/cedardb_results.csv
```

### Manual Execution
```bash
# Run individual queries manually
./build/bin/GPUDBMentalBenchmark sf1 q1
./build/bin/GPUDBMentalBenchmark sf10 q13
```

## Benchmark Scripts

The project includes automated benchmark scripts for running comprehensive performance tests:

### Data Generation
```bash
./create_tpch_data.sh
```
Generates TPC-H benchmark data at different scale factors (SF-1, SF-10). Downloads and compiles the TPC-H dbgen tool, then generates `.tbl` files in `GPUDBMentalBenchmark/Data/`.

### GPU Benchmarks
```bash
./benchmark_gpu.sh SF-1 SF-10
```
Runs all TPC-H queries (Q1, Q3, Q6, Q9, Q13) on GPU Metal implementation and saves timing results to `benchmark_results/gpu_results.csv`.

```bash
./benchmark_gpu_with_query_results.sh SF-1 SF-10
```
Extended version that also saves complete query results to `benchmark_results/gpu_logs/` for verification.

### DuckDB Benchmarks
```bash
./benchmark_duckdb.sh SF-1 SF-10
```
Runs all queries on DuckDB and saves results to `benchmark_results/duckdb_results.csv`.

```bash
./benchmark_duckdb_with_query_results.sh SF-1 SF-10
```
Extended version with query result export to `benchmark_results/duckdb_logs/`.

### CedarDB Benchmarks
```bash
# Start CedarDB container first
docker run -d --name cedardb -p 5432:5432 -e CEDAR_PASSWORD=cedar --memory=12g cedardb/cedardb

# Run benchmarks
./benchmark_cedardb.sh SF-1 SF-10
```
Runs queries on CedarDB via Docker and saves to `benchmark_results/cedardb_results.csv`.

```bash
./benchmark_cedardb_with_query_results.sh SF-1 SF-10
```
Extended version with query result logging.

## Benchmark Details

- **TPC-H Queries**: Q1 (Pricing Summary), Q3 (Shipping Priority), Q6 (Revenue Forecasting), Q9 (Product Profit), Q13 (Customer Distribution)
- **Data Format**: TPC-H standard `.tbl` files
- **Cache Strategy**: Warm cache (data pre-loaded, queries run on hot cache)
- **Timing Method**: Execution time only (excludes I/O and data loading)



