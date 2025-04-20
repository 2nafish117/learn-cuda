#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

__global__ void kernel()
{
	printf("Hello from CUDA kernel!\n");
}

__device__ void marked_device() {
	printf("hello from device\n");
}

__host__ void marked_host() {
	printf("hello from host\n");
}

__global__ void marked_both() {
	printf("hello from both\n");
}

__global__ void marked_global() {
	printf("hello from global\n");
}

int main()
{
	// running a kernel from the host
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
	
	// seeing what works
	// marked_device(); // this cannot run on host
	marked_host();
	marked_both<<<1, 1 >>>();

	marked_global<<<1, 1 >>>();

	cudaDeviceSynchronize();
	return 0;
}