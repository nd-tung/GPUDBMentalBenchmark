# Combined DuckDB vs GPU Benchmark

Generated: 2025-10-27T12:44:47.364552

Notes:
- DuckDB exec (ms) comes from DuckDB's JSON profiler when available; otherwise it falls back to wall-clock.
- DuckDB wall-clock (ms) is measured around the CLI call (commit→wait).
- GPU compute is device-only; GPU wall-clock includes submission/queue/sync; CPU merge (if any) is listed.
- Primary comparison (execution-only): DuckDB exec vs (GPU compute + CPU merge).
- Secondary (end-to-end): DuckDB wall-clock vs GPU wall-clock.

## SF-1

| Query | DuckDB exec (ms) | DuckDB wall-clock (ms) | GPU compute (ms) | GPU wall-clock (ms) | CPU merge (ms) |
|------:|------------------:|----------------------:|------------------:|--------------------:|---------------:|
| Q1 | 86.00 | 86.00 | 35.85 | 38.84 |  |
| Q3 | 60.00 | 60.00 | 11.08 | 52.95 | 1.39 |
| Q6 | 45.00 | 45.00 | 1.77 | 4.30 |  |
| Q9 | 110.00 | 110.00 | 31.45 | 54.10 |  |
| Q13 | 100.00 | 100.00 | 36.53 | 48.17 | 2.53 |

## SF-10

| Query | DuckDB exec (ms) | DuckDB wall-clock (ms) | GPU compute (ms) | GPU wall-clock (ms) | CPU merge (ms) |
|------:|------------------:|----------------------:|------------------:|--------------------:|---------------:|
| Q1 | 598.55 | 598.55 | 168.28 | 384.37 |  |
| Q3 | 317.87 | 317.87 | 45.33 | 493.88 | 21.07 |
| Q6 | 165.17 | 165.17 | 10.73 | 155.24 |  |
| Q9 | 1176.68 | 1176.68 | 390.04 | 795.72 |  |
| Q13 | 1044.42 | 1044.42 | 265.69 | 608.08 | 51.51 |

