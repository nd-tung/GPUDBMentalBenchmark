# DuckDB vs GPU Database Benchmark Results

**Generated:** Fri Oct 31 14:47:44 CET 2025
**DuckDB Version:** v1.4.1 (Andium) b390a7c376

## Test Environment
- **Hardware:** Apple M1
- **Memory:** 16 GB
- **OS:** macOS 15.7.1

## Benchmark Methodology
- **Data Loading:** Pre-loaded into memory tables (one-time cost, excluded from measurements)
- **Query Execution:** Measured only pure query execution time
- **DuckDB Settings:** Optimized with parallel processing, memory limits, and threading
- **Timing Precision:** High-precision Python timing (microsecond accuracy)

## Performance Results

The following results compare DuckDB (CPU-based analytical database) 
against the custom GPU implementation using Metal compute shaders.

### TPC-H Queries Benchmarked:
- **Q1:** Pricing Summary Report Query
- **Q3:** Shipping Priority Query  
- **Q6:** Forecasting Revenue Change Query
- **Q9:** Product Type Profit Measure Query
- **Q13:** Customer Distribution Query

### Additional Benchmarks:
- Selection queries with various selectivity
- Aggregation operations
- Hash join operations


### DuckDB Results
```
20251027_134422  SF-1   Q1                                86
20251027_134422  SF-1   Q3                                63
20251027_134422  SF-1   Q6                                46
20251027_134422  SF-1   Q9                                106
20251027_134422  SF-1   Q13                               98
20251031_144646  SF-10  Selection < 1000                  171  └──────────────┘
20251031_144646  SF-10  Selection < 10000                 85   └──────────────┘
20251031_144646  SF-10  Selection < 50000                 83   └──────────────┘
20251031_144646  SF-10  SUM(l_quantity)                   99   └─────────────────┘
20251031_144646  SF-10  Hash Join (lineitem JOIN orders)  484  └──────────────┘
20251031_144646  SF-10  TPC-H Query 1                     601  └──────────────┴──────────────┴───────────────┴──────────────────┴────────────────────┴──────────────────────┴────────────────────┴────────────────────┴─────────────────────┴─────────────┘
20251031_144646  SF-10  TPC-H Query 3                     323  └───────────────────────────────────────────────────────────┘
20251031_144646  SF-10  TPC-H Query 6                     175  └─────────────────┘
20251031_144646  SF-10  TPC-H Query 9 (standard)          891  └──────────────────────────────────────────┘
20251031_144646  SF-10  TPC-H Query 13                    824  └────────────────────┘
```
