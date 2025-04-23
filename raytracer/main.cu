#include <common/common.h>

#include <iostream>

__global__ void testKernel() {
    printf("test kernel");
}

int main() {
    int clusterSize{};
    cudaLaunchConfig_t launchConfig{};
    
    size_t dynamicSmemSize{};
    cudaError_t err1 = cudaOccupancyAvailableDynamicSMemPerBlock(&dynamicSmemSize, testKernel, 1, 1024);

    int numBlocks{};
	cudaError_t err2 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, &testKernel, 4, dynamicSmemSize);

    printf("dynamic shared mem: %zu\n", dynamicSmemSize);
    printf("num blocks: %d\n", numBlocks);
    return 0;
}