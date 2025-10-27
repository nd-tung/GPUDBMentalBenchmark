# Combined DuckDB vs GPU Benchmark

Generated: 2025-10-27T11:44:30.640471

Notes:
- DuckDB times are execution-only.
- GPU times report compute-only and wall-clock. CPU merge (if any) is shown.

## SF-1

| Query | DuckDB exec (ms) | GPU compute (ms) | GPU wall-clock (ms) | CPU merge (ms) |
|------:|------------------:|------------------:|--------------------:|---------------:|
| Q1 | 82.48 | 35.85 | 38.84 |  |
| Q3 | 59.38 | 11.08 | 52.95 | 1.39 |
| Q6 | 44.87 | 1.77 | 4.30 |  |
| Q9 | 112.73 | 31.45 | 54.10 |  |
| Q13 | 99.25 | 36.53 | 48.17 | 2.53 |

## SF-10

| Query | DuckDB exec (ms) | GPU compute (ms) | GPU wall-clock (ms) | CPU merge (ms) |
|------:|------------------:|------------------:|--------------------:|---------------:|
| Q1 | 598.55 | 168.28 | 384.37 |  |
| Q3 | 317.87 | 45.33 | 493.88 | 21.07 |
| Q6 | 165.17 | 10.73 | 155.24 |  |
| Q9 | 1176.68 | 390.04 | 795.72 |  |
| Q13 | 1044.42 | 265.69 | 608.08 | 51.51 |

