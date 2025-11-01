# Combined DuckDB vs GPU Benchmark

Generated: 2025-10-27T13:05:32.753596

Notes:
- DuckDB exec (ms) is used for comparison and comes from DuckDB's profiler when available (falls back to measured time).
- GPU compute is device-only; GPU wall-clock includes submission/queue/sync; CPU merge (if any) is listed.
- Primary comparison (execution-only): DuckDB exec vs (GPU compute + CPU merge).

## SF-1

| Query | DuckDB exec (ms) | GPU compute (ms) | GPU wall-clock (ms) | CPU merge (ms) |
|------:|------------------:|------------------:|--------------------:|---------------:|
| Q1 | 88.00 | 35.85 | 38.84 |  |
| Q3 | 59.00 | 11.08 | 52.95 | 1.39 |
| Q6 | 45.00 | 1.77 | 4.30 |  |
| Q9 | 108.00 | 31.45 | 54.10 |  |
| Q13 | 99.00 | 36.53 | 48.17 | 2.53 |

## SF-10

| Query | DuckDB exec (ms) | GPU compute (ms) | GPU wall-clock (ms) | CPU merge (ms) |
|------:|------------------:|------------------:|--------------------:|---------------:|
| Q1 | 582.00 | 168.28 | 384.37 |  |
| Q3 | 307.00 | 45.33 | 493.88 | 21.07 |
| Q6 | 176.00 | 10.73 | 155.24 |  |
| Q9 | 865.00 | 390.04 | 795.72 |  |
| Q13 | 768.00 | 265.69 | 608.08 | 51.51 |

