#!/bin/bash
# GPU Benchmark Script with Results Export
# Runs TPC-H queries Q1, Q3, Q6, Q9, Q13 and records both execution times and results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_BIN="$SCRIPT_DIR/build/bin/GPUDBMentalBenchmark"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"
LOG_DIR="$RESULTS_DIR/gpu_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GPU_CSV="$RESULTS_DIR/gpu_results.csv"

echo "=== GPU Metal TPC-H Benchmark with Results Export ==="
echo "Binary: ${BUILD_BIN}"
echo "Results CSV: ${GPU_CSV}"
echo "Results Logs: ${LOG_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Create results and log directories
mkdir -p "$RESULTS_DIR"
mkdir -p "${LOG_DIR}/${TIMESTAMP}"

# CSV header
if [[ ! -f "$GPU_CSV" ]]; then
  echo "timestamp,scale_factor,query,gpu_time_ms,wall_clock_ms,cpu_merge_ms,execution_time_gpu_cpu_ms" > "$GPU_CSV"
fi

# Ensure binary exists
if [[ ! -x "$BUILD_BIN" ]]; then
  echo "Error: GPUDBMentalBenchmark binary not found at $BUILD_BIN" >&2
  echo "Please run 'make' to build the binary first."
  exit 1
fi

run_and_capture_with_results() {
  local sf_arg=$1
  local sf_label=$2
  local out_file="${LOG_DIR}/${TIMESTAMP}/${sf_label}_full.log"
  
  echo "Running GPU benchmarks for ${sf_label}..."
  echo "  Executing: $BUILD_BIN $sf_arg (from GPUDBMentalBenchmark directory)"
  
  # Run benchmark from GPUDBMentalBenchmark directory where .metallib is located
  (cd "$SCRIPT_DIR/GPUDBMentalBenchmark" && "$BUILD_BIN" "$sf_arg") | tee "$out_file"
  
  echo ""
  echo "Extracting results for ${sf_label}..."
  
  # Parse Q1
  local q1_gpu=$(grep -E "^Total TPC-H Q1 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  local q1_wall=$(grep -E "^Total TPC-H Q1 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  local q1_cpu=$(grep -E "^Q1 CPU time:" "$out_file" | awk '{print $(NF-1)}' || echo "0.0")
  local q1_execute=$(awk -v gpu="$q1_gpu" -v cpu="$q1_cpu" 'BEGIN { if (gpu && cpu) print gpu + cpu; else if (gpu) print gpu; else print "" }')
  
  if [[ -n "$q1_gpu" ]]; then
    echo "$TIMESTAMP,$sf_label,Q1,$q1_gpu,$q1_wall,$q1_cpu,$q1_execute" >> "$GPU_CSV"
    echo "  ✓ Q1: gpu=${q1_gpu}ms, wall=${q1_wall}ms, cpu=${q1_cpu}ms, execute=${q1_execute}ms"
    
    # Extract Q1 results table to separate log
    local q1_log="${LOG_DIR}/${TIMESTAMP}/${sf_label}_Q1.log"
    echo "=== GPU Metal Q1 Results ===" > "$q1_log"
    echo "Timestamp: ${TIMESTAMP}" >> "$q1_log"
    echo "Scale Factor: ${sf_label}" >> "$q1_log"
    echo "Query: Q1" >> "$q1_log"
    echo "" >> "$q1_log"
    sed -n '/^--- Running TPC-H Query 1 Benchmark ---$/,/^Total TPC-H Q1 end-to-end:/p' "$out_file" >> "$q1_log"
  fi
  
  # Parse Q3
  local q3_gpu=$(grep -E "^Total TPC-H Q3 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  local q3_wall=$(grep -E "^Total TPC-H Q3 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  local q3_cpu=$(grep -E "^Q3 CPU time:" "$out_file" | awk '{print $(NF-1)}')
  if [[ -z "$q3_cpu" ]]; then q3_cpu=$(grep -E "^\s+CPU merge time:" "$out_file" | tail -n 1 | awk '{print $(NF-1)}' || echo "0.0"); fi
  local q3_execute=$(awk -v gpu="$q3_gpu" -v cpu="$q3_cpu" 'BEGIN { if (gpu && cpu) print gpu + cpu; else if (gpu) print gpu; else print "" }')
  
  if [[ -n "$q3_gpu" ]]; then
    echo "$TIMESTAMP,$sf_label,Q3,$q3_gpu,$q3_wall,$q3_cpu,$q3_execute" >> "$GPU_CSV"
    echo "  ✓ Q3: gpu=${q3_gpu}ms, wall=${q3_wall}ms, cpu=${q3_cpu}ms, execute=${q3_execute}ms"
    
    # Extract Q3 results
    local q3_log="${LOG_DIR}/${TIMESTAMP}/${sf_label}_Q3.log"
    echo "=== GPU Metal Q3 Results ===" > "$q3_log"
    echo "Timestamp: ${TIMESTAMP}" >> "$q3_log"
    echo "Scale Factor: ${sf_label}" >> "$q3_log"
    echo "Query: Q3" >> "$q3_log"
    echo "" >> "$q3_log"
    sed -n '/^--- Running TPC-H Query 3 Benchmark ---$/,/^Total TPC-H Q3 wall-clock:/p' "$out_file" >> "$q3_log"
  fi
  
  # Parse Q6
  local q6_gpu=$(grep -E "^Total TPC-H Q6 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  local q6_wall=$(grep -E "^Total TPC-H Q6 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  local q6_cpu=$(grep -E "^Q6 CPU time:" "$out_file" | awk '{print $(NF-1)}' || echo "0.0")
  local q6_execute=$(awk -v gpu="$q6_gpu" -v cpu="$q6_cpu" 'BEGIN { if (gpu && cpu) print gpu + cpu; else if (gpu) print gpu; else print "" }')
  
  if [[ -n "$q6_gpu" ]]; then
    echo "$TIMESTAMP,$sf_label,Q6,$q6_gpu,$q6_wall,$q6_cpu,$q6_execute" >> "$GPU_CSV"
    echo "  ✓ Q6: gpu=${q6_gpu}ms, wall=${q6_wall}ms, cpu=${q6_cpu}ms, execute=${q6_execute}ms"
    
    # Extract Q6 results
    local q6_log="${LOG_DIR}/${TIMESTAMP}/${sf_label}_Q6.log"
    echo "=== GPU Metal Q6 Results ===" > "$q6_log"
    echo "Timestamp: ${TIMESTAMP}" >> "$q6_log"
    echo "Scale Factor: ${sf_label}" >> "$q6_log"
    echo "Query: Q6" >> "$q6_log"
    echo "" >> "$q6_log"
    sed -n '/^--- Running TPC-H Query 6 Benchmark ---$/,/^Effective Bandwidth:/p' "$out_file" >> "$q6_log"
  fi
  
  # Parse Q9
  local q9_gpu=$(grep -E "^Total TPC-H Q9 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  local q9_wall=$(grep -E "^Total TPC-H Q9 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  local q9_cpu=$(grep -E "^Q9 CPU time:" "$out_file" | awk '{print $(NF-1)}' || echo "0.0")
  local q9_execute=$(awk -v gpu="$q9_gpu" -v cpu="$q9_cpu" 'BEGIN { if (gpu && cpu) print gpu + cpu; else if (gpu) print gpu; else print "" }')
  
  if [[ -n "$q9_gpu" ]]; then
    echo "$TIMESTAMP,$sf_label,Q9,$q9_gpu,$q9_wall,$q9_cpu,$q9_execute" >> "$GPU_CSV"
    echo "  ✓ Q9: gpu=${q9_gpu}ms, wall=${q9_wall}ms, cpu=${q9_cpu}ms, execute=${q9_execute}ms"
    
    # Extract Q9 results
    local q9_log="${LOG_DIR}/${TIMESTAMP}/${sf_label}_Q9.log"
    echo "=== GPU Metal Q9 Results ===" > "$q9_log"
    echo "Timestamp: ${TIMESTAMP}" >> "$q9_log"
    echo "Scale Factor: ${sf_label}" >> "$q9_log"
    echo "Query: Q9" >> "$q9_log"
    echo "" >> "$q9_log"
    sed -n '/^--- Running TPC-H Query 9 Benchmark ---$/,/^Total TPC-H Q9 wall-clock:/p' "$out_file" >> "$q9_log"
  fi
  
  # Parse Q13
  local q13_gpu=$(grep -E "^Total TPC-H Q13 GPU time:" "$out_file" | awk '{print $(NF-1)}')
  local q13_wall=$(grep -E "^Total TPC-H Q13 wall-clock:" "$out_file" | awk '{print $(NF-1)}')
  local q13_cpu=$(grep -E "^Q13 CPU time:" "$out_file" | awk '{print $(NF-1)}')
  if [[ -z "$q13_cpu" ]]; then q13_cpu=$(grep -E "^Q13 CPU merge time:" "$out_file" | awk '{print $(NF-1)}' || echo "0.0"); fi
  local q13_execute=$(awk -v gpu="$q13_gpu" -v cpu="$q13_cpu" 'BEGIN { if (gpu && cpu) print gpu + cpu; else if (gpu) print gpu; else print "" }')
  
  if [[ -n "$q13_gpu" ]]; then
    echo "$TIMESTAMP,$sf_label,Q13,$q13_gpu,$q13_wall,$q13_cpu,$q13_execute" >> "$GPU_CSV"
    echo "  ✓ Q13: gpu=${q13_gpu}ms, wall=${q13_wall}ms, cpu=${q13_cpu}ms, execute=${q13_execute}ms"
    
    # Extract Q13 results
    local q13_log="${LOG_DIR}/${TIMESTAMP}/${sf_label}_Q13.log"
    echo "=== GPU Metal Q13 Results ===" > "$q13_log"
    echo "Timestamp: ${TIMESTAMP}" >> "$q13_log"
    echo "Scale Factor: ${sf_label}" >> "$q13_log"
    echo "Query: Q13" >> "$q13_log"
    echo "" >> "$q13_log"
    sed -n '/^--- Running TPC-H Query 13 Benchmark ---$/,/^Total TPC-H Q13 wall-clock:/p' "$out_file" >> "$q13_log"
  fi
  
  echo ""
}

# Run benchmarks
run_and_capture_with_results sf1 SF-1
run_and_capture_with_results sf10 SF-10

echo ""
echo "=== Benchmark Complete ==="
echo "Results CSV: ${GPU_CSV}"
echo "Results Logs: ${LOG_DIR}/${TIMESTAMP}/"
echo ""

# Display results for this run
echo "Latest results (${TIMESTAMP}):"
grep "^${TIMESTAMP}" "${GPU_CSV}" | column -t -s,

echo ""
echo "Log files created:"
ls -lh "${LOG_DIR}/${TIMESTAMP}/"
