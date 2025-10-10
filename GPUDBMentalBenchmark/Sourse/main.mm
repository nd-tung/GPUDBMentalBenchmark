#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// --- Helper Function to Load TPC-H Data ---
std::vector<int> loadIntColumn(const std::string& filePath, int columnIndex) {
    std::vector<int> data;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return data;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::string token;
        int currentCol = 0;
        size_t start = 0;
        size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) {
                token = line.substr(start, end - start);
                data.push_back(std::stoi(token));
                break;
            }
            start = end + 1;
            end = line.find('|', start);
            currentCol++;
        }
    }
    return data;
}

// --- Function to run a single benchmark ---
void runBenchmark(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLComputePipelineState> pipelineState,
                  id<MTLBuffer> inBuffer, id<MTLBuffer> resultBuffer,
                  const std::vector<int>& cpuData, int filterValue) {
    
    // --- Encode and Dispatch Kernel ---
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
    [commandEncoder setComputePipelineState:pipelineState];
    [commandEncoder setBuffer:inBuffer offset:0 atIndex:0];
    [commandEncoder setBuffer:resultBuffer offset:0 atIndex:1];
    [commandEncoder setBytes:&filterValue length:sizeof(filterValue) atIndex:2];
    
    MTLSize gridSize = MTLSizeMake(cpuData.size(), 1, 1);
    NSUInteger threadGroupSize = [pipelineState maxTotalThreadsPerThreadgroup];
    if (threadGroupSize > cpuData.size()) { threadGroupSize = cpuData.size(); }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [commandEncoder endEncoding];
    
    // --- Execute and Wait ---
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // --- Get Timings and Print Results ---
    double gpuExecutionTime = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    double dataSizeBytes = (double)cpuData.size() * sizeof(int);
    double dataSizeGB = dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;
    
    unsigned int *resultData = (unsigned int *)resultBuffer.contents;
    unsigned int passCount = 0;
    for (size_t i = 0; i < cpuData.size(); ++i) {
        if (resultData[i] == 1) { passCount++; }
    }
    
    float selectivity = 100.0f * (float)passCount / (float)cpuData.size();
    
    NSLog(@"--- Filter Value: < %d ---", filterValue);
    NSLog(@"Selectivity: %.2f%% (%u rows matched)", selectivity, passCount);
    NSLog(@"GPU execution time: %f ms", gpuExecutionTime * 1000.0);
    NSLog(@"Effective Bandwidth: %.2f GB/s\n", bandwidth);
}


int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // --- 1. Load Data From File ---
        // Select tpch data file
        std::vector<int> cpuData = loadIntColumn("Data/SF-10/lineitem.tbl", 1);
        if (cpuData.empty()) { return 1; }
        std::cout << "Loaded " << cpuData.size() << " rows from SF-10 lineitem.tbl" << std::endl;
        
        // --- 2. Setup Metal (once) ---
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLComputePipelineState> pipelineState;
        
        @try {
            id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
            id<MTLFunction> selectionFunction = [defaultLibrary newFunctionWithName:@"selection_kernel"];
            NSError *error = nil;
            pipelineState = [device newComputePipelineStateWithFunction:selectionFunction error:&error];
            if (!pipelineState) {
                NSLog(@"Failed to create pipeline state, error: %@", error);
                return 1;
            }
        } @catch (NSException *exception) {
            NSLog(@"Error loading kernel: %@", exception.reason);
            return 1;
        }

        // --- 3. Create Buffers  ---
        const unsigned long dataSizeBytes = cpuData.size() * sizeof(int);
        id<MTLBuffer> inBuffer = [device newBufferWithBytes:cpuData.data() length:dataSizeBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> resultBuffer = [device newBufferWithLength:cpuData.size() * sizeof(unsigned int) options:MTLResourceStorageModeShared];

        // --- 4. Run Benchmarks with Different Selectivities ---
        // For l_suppkey (1 to 100,000 for SF-10)
        
        // Low Selectivity (~1%)
        runBenchmark(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 1000);
        
        // Medium Selectivity (~10%)
        runBenchmark(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 10000);
        
        // High Selectivity (~50%)
        runBenchmark(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 50000);
    }
    return 0;
}
