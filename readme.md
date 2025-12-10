# GPU SQL Query Engine

A GPU-accelerated SQL query execution engine for TPC-H benchmarks using Metal compute shaders.

## Running SQL Queries

### Basic Usage

```bash
cd GPUDBMentalBenchmark
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "YOUR_SQL_QUERY"
```

### Sample Queries

#### 1. Simple Aggregation with Predicates
```bash
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT SUM(l_extendedprice) FROM lineitem WHERE l_quantity < 24"
```

#### 2. AVG/MIN/MAX Aggregations
```bash
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT AVG(l_quantity) FROM lineitem WHERE l_quantity < 24"
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT MIN(l_discount) FROM lineitem WHERE l_quantity < 24"
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT MAX(l_tax) FROM lineitem WHERE l_quantity < 24"
```

#### 3. OR Predicates
```bash
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT SUM(l_discount) FROM lineitem WHERE l_quantity > 40 OR l_discount < 0.03"
```

#### 4. Multi-Column Predicates with Date
```bash
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT SUM(l_extendedprice) FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' AND l_discount >= 0.05 AND l_discount <= 0.07 AND l_quantity < 24"
```

#### 5. Arithmetic Expressions
```bash
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT SUM(l_extendedprice * (1 - l_discount)) FROM lineitem WHERE l_quantity < 24"
```

#### 6. JOIN Query
```bash
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT SUM(l_extendedprice) FROM lineitem JOIN orders ON l_orderkey = o_orderkey"
```

#### 7. GROUP BY
```bash
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT l_returnflag, SUM(l_quantity) FROM lineitem GROUP BY l_returnflag"
```

#### 8. ORDER BY with LIMIT
```bash
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "SELECT * FROM lineitem ORDER BY l_quantity DESC LIMIT 10"
```

### Dataset Selection

**SF-1 (default)**: ~6M rows in lineitem, 1.5M rows in orders
```bash
cd GPUDBMentalBenchmark
GPUDB_USE_GPU=1 ../build/bin/GPUDBEngineHost --sql "YOUR_QUERY"
```

**SF-10**: ~60M rows (10x larger dataset)
```bash
cd GPUDBMentalBenchmark
# Change to SF-10 directory
cd ../
mkdir -p GPUDBMentalBenchmark/Data/SF-10
# Copy your SF-10 data files to Data/SF-10/
```

## Build Instructions

```bash
make -j4
```

## Status


### Partially Working
- Multi-table JOINs (>2 tables fall back to CPU)
- Complex GROUP BY (multi-key infrastructure ready)

### Not Yet Implemented
- OUTER/LEFT/RIGHT JOINs
- Subqueries (IN/EXISTS)
- HAVING, DISTINCT
- String LIKE patterns
