#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

struct Mat {
	float* data;
	unsigned int rows, cols;
};

__host__ __device__ 
void mat_add(Mat& res, const Mat& a, const Mat& b) {
	res = {nullptr, 4, 5};
}

__global__ 
void mat_add_kernel(Mat& res, const Mat& a, const Mat& b) {
	mat_add(res, a, b);
}

int main() {
	Mat res{};
	Mat a{};
	Mat b{};

	mat_add_kernel<<<1, 1>>>(res, a, b);

	printf("%u", res.cols);

	return 0;
}