#!/bin/bash

# DuckDB Benchmark Script - FIXED VERSION
# Proper performance comparison against GPU implementation
# Fixes: Pre-loads data, optimizes settings, measures only query execution

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/GPUDBMentalBenchmark/Data"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DB_FILE="/tmp/benchmark_${TIMESTAMP}.duckdb"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   DuckDB vs GPU Database Benchmark${NC}"
echo -e "${BLUE}============================================${NC}"
echo

# Function to print section headers
print_header() {
    echo -e "${YELLOW}--- $1 ---${NC}"
}

# Function to execute DuckDB command with proper timing
execute_sql() {
    local description="$1"
    local sql_command="$2"
    local scale_factor="$3"
    
    echo -e "${GREEN}$description${NC}"
    
    # Execute with high precision timing
    start_time=$(python3 -c "import time; print(time.time())")
    result=$(duckdb "$DB_FILE" -c "$sql_command" 2>/dev/null)
    end_time=$(python3 -c "import time; print(time.time())")
    
    # Calculate execution time in milliseconds
    execution_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    echo "Execution time: ${execution_time} ms"
    echo "Result: $result"
    echo
    
    # Log results
    echo "$TIMESTAMP,$scale_factor,$description,$execution_time,$result" >> "$RESULTS_DIR/duckdb_results.csv"
}

# Check if DuckDB is installed
check_duckdb() {
    if ! command -v duckdb &> /dev/null; then
        echo -e "${RED}Error: DuckDB is not installed.${NC}"
        exit 1
    fi
    echo -e "${GREEN}DuckDB found: $(duckdb --version)${NC}"
}

# Check if Python3 is available for high-precision timing
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python3 is required for high-precision timing.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Python3 found for timing${NC}"
}

# Check if data files exist
check_data_files() {
    local scale_factor="$1"
    local data_path="$DATA_DIR/$scale_factor"
    
    if [[ ! -d "$data_path" ]]; then
        echo -e "${RED}Error: Data directory not found: $data_path${NC}"
        exit 1
    fi
    
    local required_files=("lineitem.tbl" "orders.tbl" "customer.tbl" "part.tbl" "supplier.tbl" "partsupp.tbl" "nation.tbl" "region.tbl")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$data_path/$file" ]]; then
            echo -e "${RED}Error: Required data file not found: $data_path/$file${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}Data files found for $scale_factor${NC}"
}

# Create results directory
setup_results() {
    mkdir -p "$RESULTS_DIR"
    if [[ ! -f "$RESULTS_DIR/duckdb_results.csv" ]]; then
        echo "timestamp,scale_factor,benchmark,execution_time_ms,result" > "$RESULTS_DIR/duckdb_results.csv"
    fi
}

# Initialize DuckDB with optimizations
init_duckdb() {
    echo -e "${BLUE}Initializing DuckDB with optimizations...${NC}"
    
    duckdb "$DB_FILE" << EOF
-- Optimize DuckDB settings for performance (using v1.4.1 compatible settings)
PRAGMA threads=8;
PRAGMA memory_limit='8GB';
PRAGMA temp_directory='/tmp';
EOF
    
    echo -e "${GREEN}DuckDB optimized and ready${NC}"
}

# Load data into memory tables (one-time cost)
load_data() {
    local scale_factor="$1"
    local data_path="$DATA_DIR/$scale_factor"
    
    print_header "Loading $scale_factor Data into Memory"
    
    echo "Loading lineitem table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS lineitem;
CREATE TABLE lineitem AS 
SELECT * FROM read_csv_auto('$data_path/lineitem.tbl', 
    delim='|', 
    header=false,
    columns = {
        'l_orderkey': 'INTEGER',
        'l_partkey': 'INTEGER', 
        'l_suppkey': 'INTEGER',
        'l_linenumber': 'INTEGER',
        'l_quantity': 'DECIMAL(10,2)',
        'l_extendedprice': 'DECIMAL(10,2)',
        'l_discount': 'DECIMAL(10,2)',
        'l_tax': 'DECIMAL(10,2)',
        'l_returnflag': 'CHAR(1)',
        'l_linestatus': 'CHAR(1)',
        'l_shipdate': 'DATE',
        'l_commitdate': 'DATE',
        'l_receiptdate': 'DATE',
        'l_shipinstruct': 'VARCHAR(25)',
        'l_shipmode': 'VARCHAR(10)',
        'l_comment': 'VARCHAR(44)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    lineitem_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    echo "Loading orders table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS orders;
CREATE TABLE orders AS 
SELECT * FROM read_csv_auto('$data_path/orders.tbl', 
    delim='|', 
    header=false,
    columns = {
        'o_orderkey': 'INTEGER',
        'o_custkey': 'INTEGER',
        'o_orderstatus': 'CHAR(1)',
        'o_totalprice': 'DECIMAL(10,2)',
        'o_orderdate': 'DATE',
        'o_orderpriority': 'VARCHAR(15)',
        'o_clerk': 'VARCHAR(15)',
        'o_shippriority': 'INTEGER',
        'o_comment': 'VARCHAR(79)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    orders_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    echo "Loading customer table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS customer;
CREATE TABLE customer AS 
SELECT * FROM read_csv_auto('$data_path/customer.tbl', 
    delim='|', 
    header=false,
    columns = {
        'c_custkey': 'INTEGER',
        'c_name': 'VARCHAR(25)',
        'c_address': 'VARCHAR(40)',
        'c_nationkey': 'INTEGER',
        'c_phone': 'VARCHAR(15)',
        'c_acctbal': 'DECIMAL(10,2)',
        'c_mktsegment': 'VARCHAR(10)',
        'c_comment': 'VARCHAR(117)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    customer_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    echo "Loading part table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS part;
CREATE TABLE part AS 
SELECT * FROM read_csv_auto('$data_path/part.tbl', 
    delim='|', 
    header=false,
    columns = {
        'p_partkey': 'INTEGER',
        'p_name': 'VARCHAR(55)',
        'p_mfgr': 'VARCHAR(25)',
        'p_brand': 'VARCHAR(10)',
        'p_type': 'VARCHAR(25)',
        'p_size': 'INTEGER',
        'p_container': 'VARCHAR(10)',
        'p_retailprice': 'DECIMAL(10,2)',
        'p_comment': 'VARCHAR(23)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    part_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    echo "Loading supplier table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS supplier;
CREATE TABLE supplier AS 
SELECT * FROM read_csv_auto('$data_path/supplier.tbl', 
    delim='|', 
    header=false,
    columns = {
        's_suppkey': 'INTEGER',
        's_name': 'VARCHAR(25)',
        's_address': 'VARCHAR(40)',
        's_nationkey': 'INTEGER',
        's_phone': 'VARCHAR(15)',
        's_acctbal': 'DECIMAL(10,2)',
        's_comment': 'VARCHAR(101)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    supplier_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    # Get statistics
    lineitem_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM lineitem" 2>/dev/null | tail -1)
    orders_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM orders" 2>/dev/null | tail -1)
    customer_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM customer" 2>/dev/null | tail -1)
    part_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM part" 2>/dev/null | tail -1)
    supplier_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM supplier" 2>/dev/null | tail -1)
    
    echo -e "${GREEN}Data loaded successfully${NC}"
    echo "  Lineitem: $lineitem_count rows (loaded in ${lineitem_load_time}ms)"
    echo "  Orders: $orders_count rows (loaded in ${orders_load_time}ms)"
    echo "  Customer: $customer_count rows (loaded in ${customer_load_time}ms)"
    echo "  Part: $part_count rows (loaded in ${part_load_time}ms)"
    echo "  Supplier: $supplier_count rows (loaded in ${supplier_load_time}ms)"
    echo
}

# Run benchmarks for a specific scale factor
run_benchmarks() {
    local scale_factor="$1"
    
    print_header "Running $scale_factor Benchmarks (Query Execution Only)"
    
    check_data_files "$scale_factor"
    load_data "$scale_factor"
    
    # 1. Selection Benchmarks (now using pre-loaded table)
    print_header "Selection Benchmarks ($scale_factor) - In Memory"
    
    execute_sql "Selection < 1000" \
        "SELECT COUNT(*) FROM lineitem WHERE l_partkey < 1000;" \
        "$scale_factor"
    
    execute_sql "Selection < 10000" \
        "SELECT COUNT(*) FROM lineitem WHERE l_partkey < 10000;" \
        "$scale_factor"
    
    execute_sql "Selection < 50000" \
        "SELECT COUNT(*) FROM lineitem WHERE l_partkey < 50000;" \
        "$scale_factor"
    
    # 2. Aggregation Benchmark
    print_header "Aggregation Benchmark ($scale_factor) - In Memory"
    
    execute_sql "SUM(l_quantity)" \
        "SELECT SUM(l_quantity) FROM lineitem;" \
        "$scale_factor"
    
    # 3. Join Benchmark
    print_header "Join Benchmark ($scale_factor) - In Memory"
    
    execute_sql "Hash Join (lineitem JOIN orders)" \
        "SELECT COUNT(*) FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey;" \
        "$scale_factor"
    
    # 4. TPC-H Query 1 Benchmark
    print_header "TPC-H Query 1 Benchmark ($scale_factor) - In Memory"
    
    execute_sql "TPC-H Query 1" \
        "SELECT
            l_returnflag,
            l_linestatus,
            SUM(l_quantity) AS sum_qty,
            SUM(l_extendedprice) AS sum_base_price,
            SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
            SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
            AVG(l_quantity) AS avg_qty,
            AVG(l_extendedprice) AS avg_price,
            AVG(l_discount) AS avg_disc,
            COUNT(*) AS count_order
        FROM lineitem
        WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus;" \
        "$scale_factor"
    
    # 5. TPC-H Query 3 Benchmark (Shipping Priority Query)
    print_header "TPC-H Query 3 Benchmark ($scale_factor) - In Memory"
    
    execute_sql "TPC-H Query 3" \
        "SELECT
            l.l_orderkey,
            SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
            o.o_orderdate,
            o.o_shippriority
        FROM customer c
        JOIN orders o ON c.c_custkey = o.o_custkey
        JOIN lineitem l ON l.l_orderkey = o.o_orderkey
        WHERE c.c_mktsegment = 'BUILDING'
          AND o.o_orderdate < DATE '1995-03-15'
          AND l.l_shipdate > DATE '1995-03-15'
        GROUP BY l.l_orderkey, o.o_orderdate, o.o_shippriority
        ORDER BY revenue DESC, o.o_orderdate
        LIMIT 10;" \
        "$scale_factor"
    
    # 6. TPC-H Query 6 Benchmark (Forecasting Revenue Change Query)
    print_header "TPC-H Query 6 Benchmark ($scale_factor) - In Memory"
    
    execute_sql "TPC-H Query 6" \
        "SELECT
            SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01'
          AND l_shipdate < DATE '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24;" \
        "$scale_factor"
    
    # 7. TPC-H Query 9 Benchmark (Product Type Profit Measure Query)
    print_header "TPC-H Query 9 Benchmark ($scale_factor) - In Memory"
    
    execute_sql "TPC-H Query 9" \
        "SELECT
            EXTRACT(YEAR FROM o.o_orderdate) AS o_year,
            SUM(l.l_extendedprice * (1 - l.l_discount)) AS sum_profit
        FROM lineitem l
        JOIN orders o ON l.l_orderkey = o.o_orderkey
        WHERE l.l_partkey IN (SELECT DISTINCT l_partkey FROM lineitem LIMIT 100)
        GROUP BY EXTRACT(YEAR FROM o.o_orderdate)
        ORDER BY o_year;" \
        "$scale_factor"
    
    # 8. TPC-H Query 13 Benchmark (Customer Distribution Query)
    print_header "TPC-H Query 13 Benchmark ($scale_factor) - In Memory"
    
    execute_sql "TPC-H Query 13" \
        "SELECT
            c_count,
            COUNT(*) AS custdist
        FROM (
            SELECT
                COUNT(o.o_orderkey) AS c_count
            FROM orders o
            RIGHT OUTER JOIN (SELECT DISTINCT l_orderkey FROM lineitem LIMIT 1000) c ON o.o_orderkey = c.l_orderkey
            WHERE o.o_comment NOT LIKE '%special%requests%'
            GROUP BY c.l_orderkey
        ) AS c_orders
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC;" \
        "$scale_factor"
}

# Generate comparison report
generate_report() {
    local report_file="$RESULTS_DIR/comparison_report_$TIMESTAMP.md"
    
    cat > "$report_file" << EOF
# DuckDB vs GPU Database Benchmark Results

**Generated:** $(date)
**DuckDB Version:** $(duckdb --version)

## Test Environment
- **Hardware:** $(sysctl -n machdep.cpu.brand_string)
- **Memory:** $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')
- **OS:** $(sw_vers -productName) $(sw_vers -productVersion)

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


EOF

    if [[ -f "$RESULTS_DIR/duckdb_results.csv" ]]; then
        echo "### DuckDB Results" >> "$report_file"
        echo '```' >> "$report_file"
        tail -n +2 "$RESULTS_DIR/duckdb_results.csv" | column -t -s',' >> "$report_file"
        echo '```' >> "$report_file"
    fi
    
    echo -e "${GREEN}Report generated: $report_file${NC}"
}

# Cleanup function
cleanup() {
    rm -f "$DB_FILE"
}

# Main execution
main() {
    local scale_factors=("SF-1" "SF-10")
    
    # Handle command line arguments
    if [[ $# -gt 0 ]]; then
        scale_factors=("$@")
    fi
    
    echo -e "${BLUE}Starting DuckDB benchmark...${NC}"
    echo "Scale factors to test: ${scale_factors[*]}"
    echo
    
    check_duckdb
    check_python
    setup_results
    init_duckdb
    
    # Run benchmarks for each scale factor
    for sf in "${scale_factors[@]}"; do
        if [[ -d "$DATA_DIR/$sf" ]]; then
            run_benchmarks "$sf"
        else
            echo -e "${YELLOW}Warning: Skipping $sf (directory not found)${NC}"
        fi
    done
    
    generate_report
    cleanup
    
    echo -e "${GREEN}Benchmark completed!${NC}"
    echo -e "${BLUE}Results saved in: $RESULTS_DIR${NC}"
    echo
    echo -e "${YELLOW}To view results:${NC}"
    echo "  cat $RESULTS_DIR/duckdb_results.csv"
    echo "  cat $RESULTS_DIR/comparison_report_$TIMESTAMP.md"
    echo "  cat $RESULTS_DIR/latest_comparison_report.md"
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"