# GPU Database Mental Benchmark Makefile
# Builds C++ project using metal-cpp library

# Project Configuration
PROJECT_NAME = GPUDBMentalBenchmark
SOURCE_DIR = GPUDBMentalBenchmark/Sourse
KERNEL_DIR = GPUDBMentalBenchmark/Kernels
METAL_CPP_DIR = GPUDBMentalBenchmark/metal-cpp
DATA_DIR = GPUDBMentalBenchmark/Data
BUILD_DIR = build
BIN_DIR = $(BUILD_DIR)/bin
OBJ_DIR = $(BUILD_DIR)/obj

# Compiler and flags
CXX = clang++
# macOS deployment target (adjust if needed)
MACOSX_MIN ?= 14.0
CXXFLAGS = -std=c++20 -Wall -Wextra -O3 -D_LIBCPP_DISABLE_AVAILABILITY -mmacosx-version-min=$(MACOSX_MIN)

# Include paths
INCLUDES = -I$(METAL_CPP_DIR) -IGPUDBMentalBenchmark/third_party

# Framework flags for macOS
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore

# Source files (C++ + Objective-C++ for metal-cpp wrappers)
SRC_ROOT_CPP = $(wildcard $(SOURCE_DIR)/*.cpp)
SRC_ENGINE_CPP_ALL = $(wildcard $(SOURCE_DIR)/engine/*.cpp)
SRC_ENGINE_CPP = $(filter-out $(SOURCE_DIR)/engine/host_main.cpp,$(SRC_ENGINE_CPP_ALL))
METAL_CPP_MM = $(shell find $(METAL_CPP_DIR) -name '*.mm' 2>/dev/null)
SOURCES = $(SRC_ROOT_CPP) $(SRC_ENGINE_CPP) $(METAL_CPP_MM)
OBJECTS = $(SRC_ROOT_CPP:$(SOURCE_DIR)/%.cpp=$(OBJ_DIR)/%.o) \
		  $(SRC_ENGINE_CPP:$(SOURCE_DIR)/engine/%.cpp=$(OBJ_DIR)/engine/%.o) \
		  $(METAL_CPP_MM:$(METAL_CPP_DIR)/%.mm=$(OBJ_DIR)/metal-cpp/%.o)
KERNELS = $(wildcard $(KERNEL_DIR)/*.metal)

# Target executable
TARGET = $(BIN_DIR)/$(PROJECT_NAME)
ENGINE_HOST_BIN = $(BIN_DIR)/GPUDBEngineHost
ENGINE_HOST_MAIN_SRC = $(SOURCE_DIR)/engine/host_main.cpp
ENGINE_HOST_MAIN_OBJ = $(OBJ_DIR)/engine/host_main.o
ENGINE_OBJS = $(SRC_ENGINE_CPP:$(SOURCE_DIR)/engine/%.cpp=$(OBJ_DIR)/engine/%.o) \
			  $(METAL_CPP_MM:$(METAL_CPP_DIR)/%.mm=$(OBJ_DIR)/metal-cpp/%.o)

# Metal compiler tools
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
KERNEL_AIR = $(BUILD_DIR)/kernels.air
KERNEL_AIR_OPS = $(BUILD_DIR)/kernels_ops.air
KERNEL_METALLIB = $(BUILD_DIR)/kernels.metallib

# Default target
.PHONY: all
all: $(TARGET) $(KERNEL_METALLIB) $(ENGINE_HOST_BIN)

# Create target executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	@echo "Linking $(PROJECT_NAME)..."
	$(CXX) $(OBJECTS) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

# Create engine host executable
$(ENGINE_HOST_BIN): $(ENGINE_HOST_MAIN_OBJ) $(ENGINE_OBJS) | $(BIN_DIR)
	@echo "Linking GPUDBEngineHost..."
	$(CXX) $(ENGINE_HOST_MAIN_OBJ) $(ENGINE_OBJS) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

# Build Metal kernels and place fresh metallib alongside the app assets
$(KERNEL_AIR): $(KERNEL_DIR)/DatabaseKernels.metal | $(BUILD_DIR)
	@echo "Compiling Metal kernels (.air)..."
	@$(METAL) -c $(KERNEL_DIR)/DatabaseKernels.metal -o $(KERNEL_AIR) \
		|| { echo "Metal toolchain missing; skipping DatabaseKernels.metal"; : > $(KERNEL_AIR); }

$(KERNEL_AIR_OPS): $(KERNEL_DIR)/Operators.metal | $(BUILD_DIR)
	@echo "Compiling Operators Metal (.air)..."
	@$(METAL) -c $(KERNEL_DIR)/Operators.metal -o $(KERNEL_AIR_OPS) \
		|| { echo "Metal toolchain missing; skipping Operators.metal"; : > $(KERNEL_AIR_OPS); }

$(KERNEL_METALLIB): $(KERNEL_AIR) $(KERNEL_AIR_OPS)
	@echo "Linking Metal library (.metallib)..."
	@if [ -s $(KERNEL_AIR) ]; then \
		if [ -s $(KERNEL_AIR_OPS) ]; then \
			$(METALLIB) $(KERNEL_AIR) $(KERNEL_AIR_OPS) -o $(KERNEL_METALLIB) \
				|| { echo "Metal toolchain missing; skipping metallib link"; exit 0; }; \
		else \
			$(METALLIB) $(KERNEL_AIR) -o $(KERNEL_METALLIB) \
				|| { echo "Metal toolchain missing; skipping metallib link"; exit 0; }; \
		fi; \
		cp $(KERNEL_METALLIB) GPUDBMentalBenchmark/default.metallib 2>/dev/null || true; \
	else \
		echo "No kernel AIR available; skipping metallib link"; \
	fi

# Compile source files

$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/engine/%.o: $(SOURCE_DIR)/engine/%.cpp | $(OBJ_DIR)/engine
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/metal-cpp/%.o: $(METAL_CPP_DIR)/%.mm | $(OBJ_DIR)/metal-cpp
	@echo "Compiling (ObjC++) $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -fobjc-arc -ObjC++ -c $< -o $@

# Create directories
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/engine:
	@mkdir -p $(OBJ_DIR)/engine

$(OBJ_DIR)/metal-cpp:
	@mkdir -p $(OBJ_DIR)/metal-cpp

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@echo "Clean complete"

# Install (copy to /usr/local/bin)
.PHONY: install
install: $(TARGET)
	@echo "Installing $(PROJECT_NAME)..."
	@cp $(TARGET) /usr/local/bin/
	@echo "Installation complete"

# Uninstall
.PHONY: uninstall
uninstall:
	@echo "Uninstalling $(PROJECT_NAME)..."
	@rm -f /usr/local/bin/$(PROJECT_NAME)
	@echo "Uninstall complete"

# Run the program (default: all queries with SF-10)
.PHONY: run
run: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running $(PROJECT_NAME)..."
	@cd GPUDBMentalBenchmark && ../$(TARGET)

.PHONY: run-engine-sf1
run-engine-sf1: $(ENGINE_HOST_BIN)
	@echo "Running GPUDBEngineHost with SF-1..."
	@cd GPUDBMentalBenchmark && ../$(ENGINE_HOST_BIN) sf1

.PHONY: run-engine-sf10
run-engine-sf10: $(ENGINE_HOST_BIN)
	@echo "Running GPUDBEngineHost with SF-10..."
	@cd GPUDBMentalBenchmark && ../$(ENGINE_HOST_BIN) sf10

# Run with different datasets
.PHONY: run-sf1
run-sf1: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running $(PROJECT_NAME) with SF-1 dataset..."
	@cd GPUDBMentalBenchmark && ../$(TARGET) sf1

.PHONY: run-sf10
run-sf10: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running $(PROJECT_NAME) with SF-10 dataset..."
	@cd GPUDBMentalBenchmark && ../$(TARGET) sf10

# Run TPC-H Query benchmarks individually
.PHONY: run-q1
run-q1: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 1 benchmark..."
	@cd GPUDBMentalBenchmark && ../$(TARGET) q1

.PHONY: run-q3
run-q3: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 3 benchmark..."
	@cd GPUDBMentalBenchmark && ../$(TARGET) q3

.PHONY: run-q6
run-q6: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 6 benchmark..."
	@cd GPUDBMentalBenchmark && ../$(TARGET) q6

.PHONY: run-q9
run-q9: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 9 benchmark..."
	@cd GPUDBMentalBenchmark && ../$(TARGET) q9

.PHONY: run-q13
run-q13: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running TPC-H Query 13 benchmark..."
	@cd GPUDBMentalBenchmark && ../$(TARGET) q13

# Run all TPC-H queries
.PHONY: run-all-queries
run-all-queries: $(TARGET) $(KERNEL_METALLIB)
	@echo "Running all TPC-H Query benchmarks..."
	@cd GPUDBMentalBenchmark && ../$(TARGET)

# Check if required files exist
.PHONY: check
check:
	@echo "Checking project structure..."
	@test -f $(SOURCE_DIR)/main.cpp || (echo "ERROR: main.cpp not found in $(SOURCE_DIR)" && exit 1)
	@test -d $(KERNEL_DIR) || (echo "ERROR: Kernels directory not found" && exit 1)
	@test -f $(KERNEL_DIR)/DatabaseKernels.metal || (echo "ERROR: DatabaseKernels.metal not found" && exit 1)
	@test -d $(METAL_CPP_DIR) || (echo "ERROR: metal-cpp directory not found" && exit 1)
	@test -d $(DATA_DIR) || (echo "ERROR: Data directory not found" && exit 1)
	@test -d $(DATA_DIR)/SF-1 || echo "WARNING: SF-1 dataset not found"
	@test -d $(DATA_DIR)/SF-10 || echo "WARNING: SF-10 dataset not found"
	@echo "Checking TPC-H data files..."
	@test -f $(DATA_DIR)/SF-1/lineitem.tbl || echo "WARNING: SF-1 lineitem.tbl not found"
	@test -f $(DATA_DIR)/SF-1/orders.tbl || echo "WARNING: SF-1 orders.tbl not found"
	@test -f $(DATA_DIR)/SF-1/customer.tbl || echo "WARNING: SF-1 customer.tbl not found"
	@test -f $(DATA_DIR)/SF-1/part.tbl || echo "WARNING: SF-1 part.tbl not found"
	@test -f $(DATA_DIR)/SF-1/supplier.tbl || echo "WARNING: SF-1 supplier.tbl not found"
	@test -f $(DATA_DIR)/SF-1/partsupp.tbl || echo "WARNING: SF-1 partsupp.tbl not found"
	@test -f $(DATA_DIR)/SF-1/nation.tbl || echo "WARNING: SF-1 nation.tbl not found"
	@test -f $(DATA_DIR)/SF-10/lineitem.tbl || echo "WARNING: SF-10 lineitem.tbl not found"
	@test -f $(DATA_DIR)/SF-10/orders.tbl || echo "WARNING: SF-10 orders.tbl not found"
	@test -f $(DATA_DIR)/SF-10/customer.tbl || echo "WARNING: SF-10 customer.tbl not found"
	@echo "Project structure check complete"

# Show help
.PHONY: help
help:
	@echo "GPU Database Mental Benchmark - Makefile Help"
	@echo "=============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  run-sf1           - Build and run with SF-1 dataset"
	@echo "  run-sf10          - Build and run with SF-10 dataset"
	@echo "  run-q1            - Run TPC-H Query 1 benchmark only"
	@echo "  run-q3            - Run TPC-H Query 3 benchmark only"
	@echo "  run-q6            - Run TPC-H Query 6 benchmark only"
	@echo "  run-q9            - Run TPC-H Query 9 benchmark only"
	@echo "  run-q13           - Run TPC-H Query 13 benchmark only"
	@echo "  clean             - Remove all build artifacts"
	@echo "  check             - Verify project structure"
	@echo "  help              - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make run-q1       # Run only TPC-H Query 1"
	@echo "  make run-q3       # Run only TPC-H Query 3"
	@echo "  make clean        # Clean build files"
	@echo "  make check        # Verify all files exist"

# Show project info
.PHONY: info
info:
	@echo "Project: $(PROJECT_NAME)"
	@echo "Source Directory: $(SOURCE_DIR)"
	@echo "Kernel Directory: $(KERNEL_DIR)"
	@echo "Metal-CPP Directory: $(METAL_CPP_DIR)"
	@echo "Data Directory: $(DATA_DIR)"
	@echo "Build Directory: $(BUILD_DIR)"
	@echo "Compiler: $(CXX)"
	@echo "C++ Standard: C++20"
	@echo "Frameworks: Metal, Foundation, QuartzCore"
	@echo ""
	@echo "Supported TPC-H Queries:"
	@echo "  Q1  - Pricing Summary Report Query"
	@echo "  Q3  - Shipping Priority Query"
	@echo "  Q6  - Forecasting Revenue Change Query"
	@echo "  Q9  - Product Type Profit Measure Query"
	@echo "  Q13 - Customer Distribution Query"
	@echo ""
	@echo "Available datasets: SF-1, SF-10"

# Print variables (for debugging the Makefile)
.PHONY: print-vars
print-vars:
	@echo "SOURCES: $(SOURCES)"
	@echo "OBJECTS: $(OBJECTS)"
	@echo "KERNELS: $(KERNELS)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "INCLUDES: $(INCLUDES)"
	@echo "FRAMEWORKS: $(FRAMEWORKS)"

# Force rebuild
.PHONY: rebuild
rebuild: clean all

# Compile only (no linking)
.PHONY: compile
compile: $(OBJECTS)
	@echo "Compilation complete"

# Quick test build (just compile, don't run)
.PHONY: test-build
test-build: compile
	@echo "Test build successful - source compiles without errors"

# Build target (same as all, kept for compatibility)
.PHONY: build
build: $(TARGET)

# Create a distributable package
.PHONY: package
package: $(TARGET)
	@echo "Creating distribution package..."
	@mkdir -p dist/$(PROJECT_NAME)
	@cp $(TARGET) dist/$(PROJECT_NAME)/
	@cp -r $(KERNEL_DIR) dist/$(PROJECT_NAME)/
	@cp README.md dist/$(PROJECT_NAME)/ 2>/dev/null || true
	@cp *.md dist/$(PROJECT_NAME)/ 2>/dev/null || true
	@echo "Package created in dist/$(PROJECT_NAME)/"