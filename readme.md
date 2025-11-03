# GPU Database Benchmark

GPU-accelerated database operations using Apple Metal vs DuckDB vs CedarDB comparison.

## Latest Benchmark Results

**Timestamp**: 2025-11-03  
**Methodology**: Warm cache (data pre-loaded), execution time only

### GPU Metal Results (Timestamp: 20251103_162527)

#### SF-1 Dataset (6M lineitem rows)
| Query | GPU Time (ms) | Wall Clock (ms) | CPU Merge (ms) | Execute Time (GPU+CPU) (ms) |
|-------|--------------|----------------|----------------|----------------------------|
| Q1    | 35.22        | 41.78          | 0.00           | 35.22                      |
| Q3    | 10.07        | 21.64          | 1.47           | 11.54                      |
| Q6    | 1.86         | 4.49           | 0.00           | 1.86                       |
| Q9    | 32.35        | 55.15          | 0.02           | 32.37                      |
| Q13   | 35.13        | 47.65          | 1.89           | 37.02                      |

#### SF-10 Dataset (60M lineitem rows)
| Query | GPU Time (ms) | Wall Clock (ms) | CPU Merge (ms) | Execute Time (GPU+CPU) (ms) |
|-------|--------------|----------------|----------------|----------------------------|
| Q1    | 171.67       | 350.39         | 0.00           | 171.67                     |
| Q3    | 45.41        | 546.58         | 20.25          | 65.66                      |
| Q6    | 11.14        | 164.80         | 0.00           | 11.14                      |
| Q9    | 390.81       | 876.98         | 0.14           | 390.95                     |
| Q13   | 278.96       | 483.53         | 26.43          | 305.39                     |

### DuckDB Results (Timestamp: 20251103_162622)

#### SF-1 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 89                  |
| Q3    | 61                  |
| Q6    | 46                  |
| Q9    | 113                 |
| Q13   | 102                 |

#### SF-10 Dataset
| Query | Execution Time (ms) |
|-------|---------------------|
| Q1    | 591                 |
| Q3    | 324                 |
| Q6    | 183                 |
| Q9    | 929                 |
| Q13   | 822                 |

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

### Step 1: Build GPU Benchmark
```bash
make
```

### Step 2: Run GPU Benchmark
```bash
# From GPUDBMentalBenchmark/GPUDBMentalBenchmark directory
cd GPUDBMentalBenchmark
../build/bin/GPUDBMentalBenchmark sf1   # For SF-1
../build/bin/GPUDBMentalBenchmark sf10  # For SF-10
```

### Step 3: Run DuckDB Benchmark
```bash
# From project root
./benchmark_duckdb.sh SF-1 SF-10
```

### Step 4: Start CedarDB and Run Benchmark
```bash
# Start CedarDB container
docker run -d --name cedardb -p 5432:5432 -e CEDAR_PASSWORD=cedar --memory=12g cedardb/cedardb

# Run benchmarks
./benchmark_cedardb.sh SF-1
./benchmark_cedardb.sh SF-10
```

### Step 5: View Results
```bash
# View CSV files
cat benchmark_results/gpu_results.csv
cat benchmark_results/duckdb_results.csv
cat benchmark_results/cedardb_results.csv
```

## Benchmark Details

- **TPC-H Queries**: Q1 (Pricing Summary), Q3 (Shipping Priority), Q6 (Revenue Forecasting), Q9 (Product Profit), Q13 (Customer Distribution)
- **Data Format**: TPC-H standard `.tbl` files
- **Cache Strategy**: Warm cache (data pre-loaded, queries run on hot cache)
- **Timing Method**: Execution time only (excludes I/O and data loading)



