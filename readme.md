# GPU Database Benchmark

GPU-accelerated database operations using Apple Metal vs DuckDB vs CedarDB comparison.

## Latest Benchmark Results

**Timestamp**: 2025-12-25  
**Methodology**: Warm cache (data pre-loaded), execution time only

### GPU Metal Results (Timestamp: 20251225_160158)

#### SF-1 Dataset (6M lineitem rows)
| Query | GPU Time (ms) | Wall Clock (ms) | CPU Merge (ms) | Execution Time (GPU+CPU) (ms) |
|-------|--------------|----------------|----------------|-------------------------------|
| Q1    | 33.25        | 36.16          | 0.00           | 33.25                         |
| Q3    | 10.35        | 21.54          | 1.28           | 11.63                         |
| Q6    | 1.77         | 4.49           | 0.00           | 1.77                          |
| Q9    | 21.46        | 45.53          | 0.03           | 21.49                         |
| Q13   | 30.66        | 36.88          | 1.70           | 32.36                         |

#### SF-10 Dataset (60M lineitem rows)
| Query | GPU Time (ms) | Wall Clock (ms) | CPU Merge (ms) | Execution Time (GPU+CPU) (ms) |
|-------|--------------|----------------|----------------|-------------------------------|
| Q1    | 167.57       | 326.46         | 0.01           | 167.58                        |
| Q3    | 42.35        | 387.86         | 18.42          | 60.77                         |
| Q6    | 12.26        | 109.80         | 0.00           | 12.26                         |
| Q9    | 233.24       | 779.07         | 0.19           | 233.43                        |
| Q13   | 221.60       | 315.42         | 36.51          | 258.11                        |

### DuckDB Results (Timestamp: 20251225_161546)

#### SF-1 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 53.13               |
| Q3    | 26.74               |
| Q6    | 11.66               |
| Q9    | 71.01               |
| Q13   | 64.44               |

#### SF-10 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 558.25              |
| Q3    | 258.59              |
| Q6    | 104.20              |
| Q9    | 796.00              |
| Q13   | 719.53              |

### CedarDB Results (Timestamp: 20251225_162014)

#### SF-1 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 36                  |
| Q3    | 30                  |
| Q6    | 3                   |
| Q9    | 89                  |
| Q13   | 69                  |

#### SF-10 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 332                 |
| Q3    | 614                 |
| Q6    | 35                  |
| Q9    | 1255                |
| Q13   | 2385                |

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



