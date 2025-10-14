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
CXXFLAGS = -std=c++20 -Wall -Wextra -O3
DEBUG_FLAGS = -g -DDEBUG -O0
RELEASE_FLAGS = -DNDEBUG -O3

# Include paths
INCLUDES = -I$(METAL_CPP_DIR)

# Framework flags for macOS
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore

# Source files
SOURCES = $(wildcard $(SOURCE_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SOURCE_DIR)/%.cpp=$(OBJ_DIR)/%.o)
KERNELS = $(wildcard $(KERNEL_DIR)/*.metal)

# Target executable
TARGET = $(BIN_DIR)/$(PROJECT_NAME)

# Default target
.PHONY: all
all: release

# Release build
.PHONY: release
release: CXXFLAGS += $(RELEASE_FLAGS)
release: $(TARGET)

# Debug build
.PHONY: debug
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

# Create target executable
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	@echo "Linking $(PROJECT_NAME)..."
	$(CXX) $(OBJECTS) $(FRAMEWORKS) -o $@
	@echo "Build complete: $@"

# Compile source files
$(OBJ_DIR)/%.o: $(SOURCE_DIR)/%.cpp | $(OBJ_DIR)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create directories
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

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

# Run the program
.PHONY: run
run: $(TARGET)
	@echo "Running $(PROJECT_NAME)..."
	@cd GPUDBMentalBenchmark && ../$(TARGET)

# Run with different datasets
.PHONY: run-sf1
run-sf1: $(TARGET)
	@echo "Running $(PROJECT_NAME) with SF-1 dataset..."
	@sed 's/SF-10/SF-1/g' $(SOURCE_DIR)/main.cpp > /tmp/main_sf1.cpp && \
	$(CXX) $(CXXFLAGS) $(INCLUDES) /tmp/main_sf1.cpp $(FRAMEWORKS) -o $(BIN_DIR)/$(PROJECT_NAME)_sf1 && \
	cd GPUDBMentalBenchmark && ../$(BIN_DIR)/$(PROJECT_NAME)_sf1

.PHONY: run-sf10
run-sf10: run

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
	@echo "Project structure check complete"

# Show help
.PHONY: help
help:
	@echo "GPU Database Mental Benchmark - Makefile Help"
	@echo "=============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Build release version (default)"
	@echo "  release     - Build optimized release version"
	@echo "  debug       - Build debug version with symbols"
	@echo "  clean       - Remove all build artifacts"
	@echo "  run         - Build and run the program with SF-10 data"
	@echo "  run-sf1     - Build and run with SF-1 dataset"
	@echo "  run-sf10    - Build and run with SF-10 dataset"
	@echo "  install     - Install to /usr/local/bin"
	@echo "  uninstall   - Remove from /usr/local/bin"
	@echo "  check       - Verify project structure"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build release version"
	@echo "  make debug        # Build debug version"
	@echo "  make run          # Build and run"
	@echo "  make clean        # Clean build files"

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

# Create a distributable package
.PHONY: package
package: release
	@echo "Creating distribution package..."
	@mkdir -p dist/$(PROJECT_NAME)
	@cp $(TARGET) dist/$(PROJECT_NAME)/
	@cp -r $(KERNEL_DIR) dist/$(PROJECT_NAME)/
	@cp README.md dist/$(PROJECT_NAME)/ 2>/dev/null || true
	@cp *.md dist/$(PROJECT_NAME)/ 2>/dev/null || true
	@echo "Package created in dist/$(PROJECT_NAME)/"