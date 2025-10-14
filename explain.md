# Explanation of `DatabaseKernels.metal` and `main.mm`

## `DatabaseKernels.metal`
This file contains GPU-optimized Metal kernels for various database operations. These kernels are designed to accelerate common database queries and computations using parallel processing on the GPU. Below is a summary of the key kernels:

### 1. **Selection Kernel**
- **Purpose**: Implements a filter operation (`SELECT * FROM lineitem WHERE column < filterValue`).
- **Functionality**: Each thread compares a value in the input column against a filter value and writes the result (1 for match, 0 otherwise) to an output bitmap.

### 2. **Aggregation Kernels**
- **Purpose**: Computes aggregate functions like `SUM` over a column.
- **Stages**:
  - **Stage 1**: Each thread computes a partial sum for its assigned data partition.
  - **Stage 2**: Partial sums are reduced into a final result.

### 3. **Hash Join Kernels**
- **Purpose**: Performs a hash join between two tables (`SELECT * FROM lineitem JOIN orders ON lineitem.l_orderkey = orders.o_orderkey`).
- **Phases**:
  - **Build Phase**: Constructs a hash table from the smaller table.
  - **Probe Phase**: Probes the hash table using keys from the larger table to find matches.

### 4. **TPC-H Query 1 Kernel**
- **Purpose**: Implements the TPC-H Query 1 benchmark, which involves filtering, grouping, and aggregating data.
- **Stages**:
  - **Local Aggregation**: Each threadgroup processes its own data partition and writes intermediate results.
  - **Merge**: Intermediate results are merged into a final hash table.

---

## `main.mm`
This file is the main entry point for running the database benchmarks. It uses the Metal API to load data, set up GPU buffers, and execute the kernels defined in `DatabaseKernels.metal`. Below is a summary of its key components:

### 1. **Data Loading Helpers**
- Functions to load integer, float, character, and date columns from `.tbl` files.
- Converts data into GPU-compatible formats.

### 2. **Benchmark Functions**
- **Selection Benchmark**: Tests the performance of the selection kernel with different filter values.
- **Aggregation Benchmark**: Measures the performance of the aggregation kernels.
- **Join Benchmark**: Evaluates the hash join kernels.
- **TPC-H Query 1 Benchmark**: Runs the optimized pipeline for TPC-H Query 1.

### 3. **Metal Setup**
- Initializes the Metal device, command queue, and library.
- Creates compute pipeline states for each kernel.

### 4. **Main Function**
- Loads the `.metal` library and runs all benchmarks sequentially.
- Outputs performance metrics such as GPU execution time and bandwidth.

---

This project demonstrates the use of GPU acceleration for database operations, leveraging Metal's parallel processing capabilities to achieve high performance on large datasets.