#include "effects.h"

#include <common/common.h>

namespace Effects {

__global__
void invert_kernel(
	uint8_t* imgData, 
	const int width, const int height) 
{
	const size_t pixel_index = blockDim.x * blockIdx.x + threadIdx.x;
	imgData[pixel_index] = 255 - imgData[pixel_index];
}

__global__
void blur_kernel(
	const uint8_t* __restrict__ img_data,
	uint8_t* __restrict__ blurred_img_data, 
	const int width, const int height, const int channels, const int blur_amt) {

	const uint3 pixel_index = uint3(
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y,
		0
	);

	const int diff = blur_amt;
	
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

__global__
void greyscale_kernel(
	uchar4* imgData, 
	const int width, const int height) 
{
	const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const float red 	= (float) imgData[index].x;
	const float green = (float) imgData[index].y;
	const float blue 	= (float) imgData[index].z;
	const float alpha = (float) imgData[index].w;

	const float greyscale = 0.299f * red + 0.587f * green + 0.114f * blue;
	imgData[index] = uchar4(greyscale, greyscale, greyscale, alpha);
}

void copyImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height) {
	CUDA_CHECK(cudaMemcpy2DArrayToArray(outImgData, 0, 0, inImgData, 0, 0, width * 4, height));
}

void invertImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height) {
	SCOPED_TIMER(__FUNCTION__);

	// @NOTE: cuda cannot directly modify a d3d texture via mapping and using the cudaArray, 
	// we need to use a temporary and memcpy the temporary to the array

	// @NOTE: we know the image has 4 components, rgba
	
	// copy inImgData to a temp buffer
	uint8_t* tempBuffer{};
	size_t pitch{};
	CUDA_CHECK(cudaMallocPitch(&tempBuffer, &pitch, width * 4, height));
	CUDA_CHECK(cudaMemcpy2DFromArray(tempBuffer, pitch, inImgData, 0, 0, width * 4, height, cudaMemcpyDeviceToDevice));

	// perform inversion on the tempBuffer
	dim3 work = dim3(pitch * height, 1, 1);
	dim3 numBlocks = dim3((work.x + 1023) / 1024, 1, 1);
	dim3 numThreads = dim3(1024, 1, 1);

	invert_kernel<<<numBlocks, numThreads>>>(tempBuffer, width, height);
	cudaErrorPrint(cudaGetLastError());

	// copy temp into outImgData array
	CUDA_CHECK(cudaMemcpy2DToArray(outImgData, 0, 0, tempBuffer, pitch, width * 4, height, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaFree(tempBuffer));
}

void greyscaleImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height) {
	SCOPED_TIMER(__FUNCTION__);

	// @NOTE: cuda cannot directly modify a d3d texture via mapping and using the cudaArray, 
	// we need to use a temporary and memcpy the temporary to the array

	// @NOTE: we know the image has 4 components, rgba
	
	// copy inImgData to a temp buffer
	uint8_t* tempBuffer{};
	size_t pitch{};
	CUDA_CHECK(cudaMallocPitch(&tempBuffer, &pitch, width * 4, height));
	CUDA_CHECK(cudaMemcpy2DFromArray(tempBuffer, pitch, inImgData, 0, 0, width * 4, height, cudaMemcpyDeviceToDevice));

	// perform inversion on the tempBuffer
	dim3 work = dim3((pitch / 4) * height, 1, 1);
	dim3 numBlocks = dim3((work.x + 1023) / 1024, 1, 1);
	dim3 numThreads = dim3(1024, 1, 1);

	greyscale_kernel<<<numBlocks, numThreads>>>((uchar4*) tempBuffer, width, height);
	cudaErrorPrint(cudaGetLastError());

	// copy temp into outImgData array
	CUDA_CHECK(cudaMemcpy2DToArray(outImgData, 0, 0, tempBuffer, pitch, width * 4, height, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaFree(tempBuffer));
}

// @TODO: blur params
// @TODO: try blur with custom kernel?
void blurImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height, const BlurParams& params) {
	SCOPED_TIMER(__FUNCTION__);

}

// @TODO: sobel params
void sobelImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height, const SobelParams& params) {
	SCOPED_TIMER(__FUNCTION__);
}

// void blur_img(uint8_t* img_data, int width, int height, const int channels, int blur_amt)
// {
// 	SCOPED_TIMER(__FUNCTION__);
// 	const size_t size = width * height * channels;

// 	uint8_t* device_img_data = nullptr;
// 	CUDA_CHECK(cudaMalloc(&device_img_data, size));
// 	CUDA_CHECK(cudaMemcpy(device_img_data, img_data, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
	
// 	uint8_t* device_blurred_img_data = nullptr;
// 	CUDA_CHECK(cudaMalloc(&device_blurred_img_data, size));

// 	dim3 work = dim3(width, height, 1);
// 	dim3 numBlocks = dim3((width + 31) / 32, (height + 31) / 32, 1);
// 	dim3 numThreads = dim3(32, 32, 1);
// 	blur_kernel<<<numBlocks, numThreads>>>(device_img_data, device_blurred_img_data, width, height, channels, blur_amt);
// 	cudaErrorPrint(cudaGetLastError());

// 	CUDA_CHECK(cudaMemcpy(img_data, device_blurred_img_data, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
	
// 	CUDA_CHECK(cudaFree(device_img_data));
// 	CUDA_CHECK(cudaFree(device_blurred_img_data));
// }

} // namespace Effects