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

__global__ void saxpy(int n, float a, float* x, float* y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n) {
		y[i] = a * x[i] + y[i];
	}
}

int main()
{
	// running a kernel from the host
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

	// can also call the kernel like this
	cudaLaunchKernel(&kernel, dim3{1, 1, 1}, dim3{1, 1, 1}, nullptr);
	
	// seeing what works
	// marked_device(); // this cannot run on host
	marked_host();
	marked_both<<<1, 1 >>>();

	marked_global<<<1, 1 >>>();

	{
		cudaError_t err = cudaDeviceSynchronize();
		printf("cuda err: %d\n", err);
	}

	// saxpy loop
	{
		int n = 10;
		float a = 1;

		float x[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
		float y[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

		float* dx{};
		cudaError_t err1 = cudaMalloc(&dx, n * sizeof(float));
		cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		float* dy{};
		cudaError_t err2 = cudaMalloc(&dy, n * sizeof(float));
		cudaMemcpy(dy, y, n * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		for(int i = 0; i < n; ++i) {
			printf("%f ", y[i]);
		}
		printf("\n");
	
		saxpy<<<1, n>>>(n, a, dx, dy);
		
		cudaMemcpy(y, dy, n * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		cudaFree(dx);
		dx = nullptr;
		cudaFree(dy);
		dy = nullptr;
		
		for(int i = 0; i < n; ++i) {
			printf("%f ", y[i]);
		}
		printf("\n");
	}

	{
		cudaError_t err = cudaDeviceSynchronize();
		printf("cuda err: %d\n", err);
	}

	return 0;
}