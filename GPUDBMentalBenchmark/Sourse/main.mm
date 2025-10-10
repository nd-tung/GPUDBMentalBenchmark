#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// --- Helper to Load Integer Column ---
std::vector<int> loadIntColumn(const std::string& filePath, int columnIndex) {
    std::vector<int> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) { token = line.substr(start, end - start); data.push_back(std::stoi(token)); break; }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

// --- Helper to Load Float Column ---
std::vector<float> loadFloatColumn(const std::string& filePath, int columnIndex) {
    std::vector<float> data;
    std::ifstream file(filePath);
    if (!file.is_open()) { std::cerr << "Error: Could not open file " << filePath << std::endl; return data; }
    std::string line;
    while (std::getline(file, line)) {
        std::string token; int currentCol = 0; size_t start = 0; size_t end = line.find('|');
        while (end != std::string::npos) {
            if (currentCol == columnIndex) { token = line.substr(start, end - start); data.push_back(std::stof(token)); break; }
            start = end + 1; end = line.find('|', start); currentCol++;
        }
    }
    return data;
}

// --- Selection Benchmark Test Function ---
void runSingleSelectionTest(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLComputePipelineState> pipelineState,
                            id<MTLBuffer> inBuffer, id<MTLBuffer> resultBuffer,
                            const std::vector<int>& cpuData, int filterValue) {
    
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
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    double gpuExecutionTime = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    double dataSizeBytes = (double)cpuData.size() * sizeof(int);
    double dataSizeGB = dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;
    
    unsigned int *resultData = (unsigned int *)resultBuffer.contents;
    unsigned int passCount = 0;
    for (size_t i = 0; i < cpuData.size(); ++i) { if (resultData[i] == 1) { passCount++; } }
    float selectivity = 100.0f * (float)passCount / (float)cpuData.size();
    
    NSLog(@"--- Filter Value: < %d ---", filterValue);
    NSLog(@"Selectivity: %.2f%% (%u rows matched)", selectivity, passCount);
    NSLog(@"GPU execution time: %f ms", gpuExecutionTime * 1000.0);
    NSLog(@"Effective Bandwidth: %.2f GB/s\n", bandwidth);
}

// --- Main Function for Selection Benchmark ---
void runSelectionBenchmark(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLLibrary> library) {
    NSLog(@"--- Running Selection Benchmark ---");

    //Select tpch data file
    std::vector<int> cpuData = loadIntColumn("Data/SF-10/lineitem.tbl", 1);
    if (cpuData.empty()) { return; }
    NSLog(@"Loaded %lu rows for selection.", cpuData.size());

    NSError *error = nil;
    id<MTLFunction> selectionFunction = [library newFunctionWithName:@"selection_kernel"];
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:selectionFunction error:&error];
    if (!pipelineState) { NSLog(@"Failed to create selection pipeline state: %@", error); return; }

    const unsigned long dataSizeBytes = cpuData.size() * sizeof(int);
    id<MTLBuffer> inBuffer = [device newBufferWithBytes:cpuData.data() length:dataSizeBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultBuffer = [device newBufferWithLength:cpuData.size() * sizeof(unsigned int) options:MTLResourceStorageModeShared];

    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 1000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 10000);
    runSingleSelectionTest(device, commandQueue, pipelineState, inBuffer, resultBuffer, cpuData, 50000);
}


// --- Main Function for Aggregation Benchmark ---
void runAggregationBenchmark(id<MTLDevice> device, id<MTLCommandQueue> commandQueue, id<MTLLibrary> library) {
    NSLog(@"--- Running Aggregation Benchmark ---");

    //Select tpch data file
    std::vector<float> cpuData = loadFloatColumn("Data/SF-10/lineitem.tbl", 4);
    if (cpuData.empty()) return;
    NSLog(@"Loaded %lu rows for aggregation.", cpuData.size());
    const unsigned long dataSizeBytes = cpuData.size() * sizeof(float);
    uint dataSize = (uint)cpuData.size(); // The actual number of elements

    NSError *error = nil;
    id<MTLFunction> stage1Function = [library newFunctionWithName:@"sum_kernel_stage1"];
    id<MTLComputePipelineState> stage1Pipeline = [device newComputePipelineStateWithFunction:stage1Function error:&error];
    if (!stage1Pipeline) { NSLog(@"Failed to create stage 1 pipeline state: %@", error); return; }

    id<MTLFunction> stage2Function = [library newFunctionWithName:@"sum_kernel_stage2"];
    id<MTLComputePipelineState> stage2Pipeline = [device newComputePipelineStateWithFunction:stage2Function error:&error];
    if (!stage2Pipeline) { NSLog(@"Failed to create stage 2 pipeline state: %@", error); return; }

    const int numThreadgroups = 2048;
    id<MTLBuffer> inBuffer = [device newBufferWithBytes:cpuData.data() length:dataSizeBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> partialSumsBuffer = [device newBufferWithLength:numThreadgroups * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> resultBuffer = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    
    id<MTLComputeCommandEncoder> stage1Encoder = [commandBuffer computeCommandEncoder];
    [stage1Encoder setComputePipelineState:stage1Pipeline];
    [stage1Encoder setBuffer:inBuffer offset:0 atIndex:0];

    [stage1Encoder setBuffer:partialSumsBuffer offset:0 atIndex:1];
    [stage1Encoder setBytes:&dataSize length:sizeof(dataSize) atIndex:2]; // <-- THIS IS THE NEW LINE
    
    NSUInteger stage1ThreadGroupSize = [stage1Pipeline maxTotalThreadsPerThreadgroup];
    MTLSize stage1GridSize = MTLSizeMake(numThreadgroups, 1, 1);
    MTLSize stage1GroupSize = MTLSizeMake(stage1ThreadGroupSize, 1, 1);
    [stage1Encoder dispatchThreadgroups:stage1GridSize threadsPerThreadgroup:stage1GroupSize];
    [stage1Encoder endEncoding];

    id<MTLComputeCommandEncoder> stage2Encoder = [commandBuffer computeCommandEncoder];
    [stage2Encoder setComputePipelineState:stage2Pipeline];
    [stage2Encoder setBuffer:partialSumsBuffer offset:0 atIndex:0];
    [stage2Encoder setBuffer:resultBuffer offset:0 atIndex:1];
    [stage2Encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [stage2Encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    double gpuExecutionTime = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    double dataSizeGB = (double)dataSizeBytes / (1024.0 * 1024.0 * 1024.0);
    double bandwidth = dataSizeGB / gpuExecutionTime;

    float *finalSum = (float *)resultBuffer.contents;
    NSLog(@"Final SUM(l_quantity): %.2f", finalSum[0]);
    NSLog(@"GPU execution time: %f ms", gpuExecutionTime * 1000.0);
    NSLog(@"Effective Bandwidth: %.2f GB/s\n", bandwidth);
}


// --- Main Entry Point ---
int main(int argc, const char * argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLLibrary> library;
        @try {
            library = [device newDefaultLibrary];
        } @catch (NSException *exception) {
            NSLog(@"Error loading .metal library: %@", exception.reason);
            return 1;
        }

        // Run both benchmarks
        runSelectionBenchmark(device, commandQueue, library);
        runAggregationBenchmark(device, commandQueue, library);
    }
    return 0;
}
