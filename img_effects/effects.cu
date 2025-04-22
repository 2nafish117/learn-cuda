#include <common/common.cuh>

namespace Effects {

__global__
void invert_kernel(
	uint8_t* img_data, 
	const int width, const int height, const int channels) 
{
	const size_t pixel_index = blockDim.x * blockIdx.x + threadIdx.x;

	for(int i = 0; i < channels; ++i) {
		const size_t comp_index = channels * pixel_index + i;
		uint8_t comp_val = img_data[comp_index];
		img_data[comp_index] = 255 - comp_val;
	}
}

__global__
void blur_kernel(
	const uint8_t* __restrict__ img_data,
	uint8_t* __restrict__ blurred_img_data, 
	const int width, const int height, const int channels) {

	const uint3 pixel_index = uint3(
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y,
		0
	);

	const int diff = 8;
	
	for(int c = 0; c < channels; ++c) {

		float sum = 0.0f;
		for(int dx = -diff; dx <= diff; ++dx) {
			for(int dy = -diff; dy <= diff; ++dy) {
				const int3 pix = int3(pixel_index.x + dx, pixel_index.y + dy, 0);
				if(pix.x >= 0 && pix.x < width && pix.y >= 0 && pix.y < height) {
					size_t channel_index = channels * (pix.x + width * pix.y) + c;
					sum += (float) img_data[channel_index] / 255.0f;
				}
			}
		}
	
		float denom = (2 * diff + 1) * (2 * diff + 1);
		blurred_img_data[channels * (pixel_index.x + pixel_index.y * width) + c] = (uint8_t) ((sum / denom) * 255.0f);
	}
}

void invert_img(
	uint8_t* img_data, 
	int width, int height, const int channels) 
{
	SCOPED_TIMER(__FUNCTION__);

	const size_t size = width * height * channels;
	uint8_t* device_img_data = nullptr;

	CUDA_CHECK(cudaMalloc(&device_img_data, size));
	CUDA_CHECK(cudaMemcpy(device_img_data, img_data, size, cudaMemcpyKind::cudaMemcpyHostToDevice));

	dim3 work = dim3(width * height, 1, 1);
	dim3 numBlocks = dim3((work.x + 1023) / 1024, 1, 1);
	dim3 numThreads = dim3(1024, 1, 1);

	// @TODO: get device pointer here???
	invert_kernel<<<numBlocks, numThreads>>>(device_img_data, width, height, channels);
	cudaErrorPrint(cudaGetLastError());

	CUDA_CHECK(cudaMemcpy(img_data, device_img_data, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(device_img_data));
}

void blur_img(uint8_t* img_data, int width, int height, const int channels)
{
	SCOPED_TIMER(__FUNCTION__);
	const size_t size = width * height * channels;

	uint8_t* device_img_data = nullptr;
	CUDA_CHECK(cudaMalloc(&device_img_data, size));
	CUDA_CHECK(cudaMemcpy(device_img_data, img_data, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
	
	uint8_t* device_blurred_img_data = nullptr;
	CUDA_CHECK(cudaMalloc(&device_blurred_img_data, size));

	dim3 work = dim3(width, height, 1);
	dim3 numBlocks = dim3((width + 31) / 32, (height + 31) / 32, 1);
	dim3 numThreads = dim3(32, 32, 1);
	blur_kernel<<<numBlocks, numThreads>>>(device_img_data, device_blurred_img_data, width, height, channels);
	cudaErrorPrint(cudaGetLastError());

	CUDA_CHECK(cudaMemcpy(img_data, device_blurred_img_data, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
	
	CUDA_CHECK(cudaFree(device_img_data));
	CUDA_CHECK(cudaFree(device_blurred_img_data));
}


} // namespace Effects