#!/usr/bin/env python3
import csv
import os
import sys
from datetime import datetime

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(repo_dir, 'results')

def load_duckdb_results(path):
    import re
    rows = []
    # Supports both legacy (exec only) and new (exec, wall) formats:
    # Legacy: timestamp,sf,bench,exec_ms,result
    # New:    timestamp,sf,bench,exec_ms,wall_ms,result
    line_re = re.compile(r"^(\d{8}_\d{6}),(SF-1|SF-10),(.*?),(\d+\.?\d*)(?:,(\d+\.?\d*))?,")
    with open(path, 'r') as f:
        for line in f:
            m = line_re.match(line.strip())
            if not m:
                continue
            ts, sf, bench, exec_ms, wall_ms = m.groups()
            key = None
            if bench == 'TPC-H Query 1': key = 'Q1'
            elif bench == 'TPC-H Query 3': key = 'Q3'
            elif bench == 'TPC-H Query 6': key = 'Q6'
            elif bench.startswith('TPC-H Query 9'): key = 'Q9'
            elif bench == 'TPC-H Query 13': key = 'Q13'
            if key:
                exec_val = float(exec_ms)
                wall_val = float(wall_ms) if wall_ms is not None else exec_val
                rows.append({
                    'timestamp': ts,
                    'scale_factor': sf,
                    'query': key,
                    'duckdb_time_ms': exec_val,
                    'duckdb_wall_ms': wall_val,
                })
    return rows

def load_gpu_results(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'timestamp': r['timestamp'],
                'scale_factor': r['scale_factor'],
                'query': r['query'],
                'gpu_time_ms': float(r['gpu_time_ms']) if r['gpu_time_ms'] else None,
                'wall_clock_ms': float(r['wall_clock_ms']) if r['wall_clock_ms'] else None,
                'cpu_merge_ms': float(r['cpu_merge_ms']) if r['cpu_merge_ms'] else None,
            })
    return rows

def generate_report(duckdb_csv, gpu_csv, out_path):
    duck = load_duckdb_results(duckdb_csv)
    gpu = load_gpu_results(gpu_csv)

    # Index by (sf, query)
    duck_idx = {(r['scale_factor'], r['query']): r for r in duck}
    gpu_idx = {(r['scale_factor'], r['query']): r for r in gpu}

    sfs = ['SF-1', 'SF-10']
    queries = ['Q1', 'Q3', 'Q6', 'Q9', 'Q13']

    with open(out_path, 'w') as out:
        out.write(f"# Combined DuckDB vs GPU Benchmark\n\n")
        out.write(f"Generated: {datetime.now().isoformat()}\n\n")
        out.write("Notes:\n")
        out.write("- DuckDB exec (ms) is used for comparison and comes from DuckDB's profiler when available (falls back to measured time).\n")
        out.write("- GPU compute is device-only; GPU wall-clock includes submission/queue/sync; CPU merge (if any) is listed.\n")
        out.write("- Primary comparison (execution-only): DuckDB exec vs (GPU compute + CPU merge).\n\n")
        for sf in sfs:
            out.write(f"## {sf}\n\n")
            out.write("| Query | DuckDB exec (ms) | GPU compute (ms) | GPU wall-clock (ms) | CPU merge (ms) |\n")
            out.write("|------:|------------------:|------------------:|--------------------:|---------------:|\n")
            for q in queries:
                d = duck_idx.get((sf, q))
                g = gpu_idx.get((sf, q))
                out.write("| {q} | {d_ms} | {g_gpu} | {g_wall} | {g_cpu} |\n".format(
                    q=q,
                    d_ms=f"{d['duckdb_time_ms']:.2f}" if d else "",
                    g_gpu=f"{g['gpu_time_ms']:.2f}" if g and g['gpu_time_ms'] is not None else "",
                    g_wall=f"{g['wall_clock_ms']:.2f}" if g and g['wall_clock_ms'] is not None else "",
                    g_cpu=f"{g['cpu_merge_ms']:.2f}" if g and g['cpu_merge_ms'] is not None else "",
                ))
            out.write("\n")

if __name__ == '__main__':
    duckdb_csv = os.path.join(results_dir, 'duckdb_results.csv')
    gpu_csv = os.path.join(results_dir, 'gpu_results.csv')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_md = os.path.join(results_dir, f'comparison_report_{ts}_combined.md')

    if not os.path.exists(duckdb_csv):
        print(f"Error: DuckDB results not found: {duckdb_csv}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(gpu_csv):
        print(f"Error: GPU results not found: {gpu_csv}", file=sys.stderr)
        sys.exit(1)

    generate_report(duckdb_csv, gpu_csv, out_md)
    print(f"Combined report generated: {out_md}")
