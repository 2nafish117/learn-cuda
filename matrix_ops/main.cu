#include <common/common.h>

#include <iostream>
#include <chrono>
#include <random>
#include <cassert>

struct Mat {
	float* data;
	float* device_data;
	uint32_t rows, cols;

	__host__ __device__
	float get(uint32_t r, uint32_t c) const {
		size_t index = r + c * rows;
		assert(index < count());

#if __CUDACC__
		return device_data[r + c * rows];
#else
		return data[r + c * rows];
#endif
	}

	__host__ __device__
	float& at(uint32_t r, uint32_t c) {
		size_t index = r + c * rows;
		assert(index < count());
#if __CUDACC__
		return device_data[r + c * rows];
#else
		return data[r + c * rows];
#endif
	}

	__host__ __device__
	size_t count() const {
		return rows * cols;
	}
};

Mat mat_make(uint32_t rows, uint32_t cols) {
	Mat m;
	m.rows = rows;
	m.cols = cols;
	m.data = (float*) malloc(sizeof(float) * m.count());
	CUDA_CHECK(cudaMalloc(&m.device_data, sizeof(float) * m.count()));
	return m;
}

void mat_delete(Mat& m) {
	free(m.data);
	CUDA_CHECK(cudaFree(m.device_data));
	m.rows = m.cols = 0;
}

void mat_fill(Mat& m, float val) {
	for(int i = 0; i < m.count(); ++i) {
		m.data[i] = val;
	}

	CUDA_CHECK(cudaMemcpy(m.device_data, m.data, sizeof(float) * m.count(), cudaMemcpyKind::cudaMemcpyHostToDevice));
}

void init_random(Mat& m) {
	using namespace std;
	mt19937 engine = mt19937(0);
    uniform_real_distribution<float> generator = uniform_real_distribution<float>(0, 1);

	for(int i = 0; i < m.count(); ++i) {
		float random_val = generator(engine);
		m.data[i] = random_val;
	}

	CUDA_CHECK(cudaMemcpy(m.device_data, m.data, sizeof(float) * m.count(), cudaMemcpyKind::cudaMemcpyHostToDevice));
}

void mat_add(Mat& res, const Mat& a, const Mat& b) {
	// @TODO: check if we can add

	for(int i = 0; i < res.count(); ++i) {
		res.data[i] = a.data[i] + b.data[i];
	}
}

void mat_mul(Mat& res, const Mat& a, const Mat& b) {
	// @TODO: check if we can mul

	for(int i = 0; i < res.rows; ++i) {
		for(int j = 0; j < res.cols; ++j) {
			float sum = 0.0f;

			for(int k = 0; k < a.cols; ++k) {
				sum += a.get(i, k) * b.get(k, j);
			}

			res.at(i, j) = sum;
		}
	}
}

// @NOTE: you cannot pass variables by reference in cuda, only values!!
__global__ 
void mat_add_kernel(Mat res, const Mat a, const Mat b) {
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index >= res.count()) {
		return;
	}

	res.device_data[index] = a.device_data[index] + b.device_data[index];
}

__global__ 
void mat_mul_kernel(Mat res, const Mat a, const Mat b) {
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i >= res.rows || j >= res.cols) {
		return;
	}

	float sum = 0.0f;
	for(int k = 0; k < a.cols; ++k) {
		sum += a.get(i, k) * b.get(k, j);
	}

	res.at(i, j) = sum;
}

void mat_add_cuda(Mat& res, const Mat& a, const Mat& b) {
	// @TODO: check if we can add

	dim3 work = dim3(res.count(), 1, 1);
	dim3 nb = dim3((work.x + 1023) / 1024, 1, 1);
	dim3 bd = dim3(1024, 1, 1);

	mat_add_kernel<<<nb, bd>>>(res, a, b);
	cudaErrorPrint(cudaGetLastError());

	CUDA_CHECK(cudaMemcpy(res.data, res.device_data, sizeof(float) * res.count(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

void mat_mul_cuda(Mat& res, const Mat& a, const Mat& b) {
	// @TODO: check if we can mul

	dim3 work = dim3(res.rows, res.cols, 1);
	dim3 nb = dim3((work.x + 31) / 32, (work.y + 31) / 32, work.z);
	dim3 bd = dim3(32, 32, 1);

	mat_mul_kernel<<<nb, bd>>>(res, a, b);
	cudaErrorPrint(cudaGetLastError());

	CUDA_CHECK(cudaMemcpy(res.data, res.device_data, sizeof(float) * res.count(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
}

int main() {
	Mat res = mat_make(5, 5);
	
	Mat a = mat_make(5, 2);
	mat_fill(a, 1);
	Mat b = mat_make(2, 5);
	mat_fill(b, 2);

	// {
	// 	SCOPED_TIMER("mat_add");
	// 	mat_add(res, a, b);
	// }

	// {
	// 	SCOPED_TIMER("mat_add_cuda");
	// 	mat_add_cuda(res, a, b);
	// }

	// {
	// 	SCOPED_TIMER("mat_mul");
	// 	mat_mul(res, a, b);
	// }

	{
		SCOPED_TIMER("mat_mul_cuda");
		mat_mul_cuda(res, a, b);
	}

	mat_delete(res);
	mat_delete(a);
	mat_delete(b);

	return 0;
}