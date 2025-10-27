#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BIN="$SCRIPT_DIR/build/bin/GPUDBMentalBenchmark"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$RESULTS_DIR"
GPU_CSV="$RESULTS_DIR/gpu_results.csv"

# CSV header
if [[ ! -f "$GPU_CSV" ]]; then
  echo "timestamp,scale_factor,query,gpu_time_ms,wall_clock_ms,cpu_merge_ms" > "$GPU_CSV"
fi

run_and_capture() {
  local sf_arg=$1
  local sf_label=$2
  local out_file="$RESULTS_DIR/gpu_${sf_label}_${TIMESTAMP}.log"
  echo "Running GPU benchmarks for ${sf_label}..."
  "$BUILD_BIN" "$sf_arg" | tee "$out_file" >/dev/null

  # Parse standardized lines
  # Q1
  q1_gpu=$(grep -E "^Total TPC-H Q1 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  q1_wall=$(grep -E "^Total TPC-H Q1 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  if [[ -n "$q1_gpu" ]]; then echo "$TIMESTAMP,$sf_label,Q1,$q1_gpu,$q1_wall," >> "$GPU_CSV"; fi

  # Q3
  q3_gpu=$(grep -E "^Total TPC-H Q3 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  q3_wall=$(grep -E "^Total TPC-H Q3 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  q3_cpu=$(grep -E "^  CPU merge time:" "$out_file" | tail -n 1 | awk '{print $(NF-1)}')
  if [[ -n "$q3_gpu" ]]; then echo "$TIMESTAMP,$sf_label,Q3,$q3_gpu,$q3_wall,$q3_cpu" >> "$GPU_CSV"; fi

  # Q6
  q6_gpu=$(grep -E "^Total TPC-H Q6 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  q6_wall=$(grep -E "^Total TPC-H Q6 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  if [[ -n "$q6_gpu" ]]; then echo "$TIMESTAMP,$sf_label,Q6,$q6_gpu,$q6_wall," >> "$GPU_CSV"; fi

  # Q9
  q9_gpu=$(grep -E "^Total TPC-H Q9 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  q9_wall=$(grep -E "^Total TPC-H Q9 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  if [[ -n "$q9_gpu" ]]; then echo "$TIMESTAMP,$sf_label,Q9,$q9_gpu,$q9_wall," >> "$GPU_CSV"; fi

  # Q13
  q13_gpu=$(grep -E "^Total TPC-H Q13 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  q13_wall=$(grep -E "^Total TPC-H Q13 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  q13_cpu=$(grep -E "^Q13 CPU merge time:" "$out_file" | awk '{print $(NF-1)}')
  if [[ -n "$q13_gpu" ]]; then echo "$TIMESTAMP,$sf_label,Q13,$q13_gpu,$q13_wall,$q13_cpu" >> "$GPU_CSV"; fi
}

# Ensure binary exists
if [[ ! -x "$BUILD_BIN" ]]; then
  echo "Error: GPUDBMentalBenchmark binary not found at $BUILD_BIN" >&2
  exit 1
fi

run_and_capture sf1 SF-1
run_and_capture sf10 SF-10 || true

echo "GPU results saved to: $GPU_CSV"
