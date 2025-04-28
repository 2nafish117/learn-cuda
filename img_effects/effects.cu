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
void greyscale_kernel(
	uchar4* imgData, 
	const int width, const int height) 
{
	const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const float red 	= (float) imgData[index].x;
	const float green 	= (float) imgData[index].y;
	const float blue 	= (float) imgData[index].z;
	const float alpha 	= (float) imgData[index].w;

	const float greyscale = 0.299f * red + 0.587f * green + 0.114f * blue;
	imgData[index] = uchar4(greyscale, greyscale, greyscale, alpha);
}

__global__
void blur_kernel(
	const uchar4* __restrict__ inImgData,
	uchar4* __restrict__ outImgData, 
	const int width, const int height, const int pitch, const int2 blurSize) {

	const uint3 pixelIndex = uint3(
		blockDim.x * blockIdx.x + threadIdx.x,
		blockDim.y * blockIdx.y + threadIdx.y,
		0
	);

	const int2 diff = int2(blurSize.x / 2, blurSize.y / 2);
	const int pitchWidth = pitch / 4;

	float3 sum = float3(0.0f, 0.0f, 0.0f);
	for(int dx = -diff.x; dx <= diff.x; ++dx) {
		for(int dy = -diff.y; dy <= diff.y; ++dy) {
			const int3 pix = int3(pixelIndex.x + dx, pixelIndex.y + dy, 0);

			if(pix.x >= 0 && pix.x < width && pix.y >= 0 && pix.y < height) {
				const size_t linearIndex = pix.x + pitchWidth * pix.y;
				const uchar4 data = inImgData[linearIndex];
				sum = float3(sum.x + data.x, sum.y + data.y, sum.z + data.z);
			}
		}
	}

	const float denom = blurSize.x * blurSize.y;
	const size_t pixelLinearIndex = pixelIndex.x + pixelIndex.y * pitchWidth;
	const float alpha = inImgData[pixelLinearIndex].w;
	outImgData[pixelLinearIndex] = uchar4(sum.x / denom, sum.y / denom, sum.z / denom, alpha);
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
	
	uint8_t* tempBuffer{};
	size_t pitch{};
	CUDA_CHECK(cudaMallocPitch(&tempBuffer, &pitch, width * 4, height));
	CUDA_CHECK(cudaMemcpy2DFromArray(tempBuffer, pitch, inImgData, 0, 0, width * 4, height, cudaMemcpyDeviceToDevice));

	dim3 work = dim3((pitch / 4) * height, 1, 1);
	dim3 numBlocks = dim3((work.x + 1023) / 1024, 1, 1);
	dim3 numThreads = dim3(1024, 1, 1);

	greyscale_kernel<<<numBlocks, numThreads>>>((uchar4*) tempBuffer, width, height);
	cudaErrorPrint(cudaGetLastError());

	CUDA_CHECK(cudaMemcpy2DToArray(outImgData, 0, 0, tempBuffer, pitch, width * 4, height, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaFree(tempBuffer));
}

// @TODO: blur params
// @TODO: try blur with custom kernel?
void blurImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height, const BlurParams& params) {
	SCOPED_TIMER(__FUNCTION__);

	// @NOTE: cuda cannot directly modify a d3d texture via mapping and using the cudaArray, 
	// we need to use a temporary and memcpy the temporary to the array

	// @NOTE: we know the image has 4 components, rgba
	
	uchar4* tempInputBuffer{};
	size_t pitchInput{};
	CUDA_CHECK(cudaMallocPitch(&tempInputBuffer, &pitchInput, width * 4, height));
	CUDA_CHECK(cudaMemcpy2DFromArray(tempInputBuffer, pitchInput, inImgData, 0, 0, width * 4, height, cudaMemcpyDeviceToDevice));

	uchar4* tempOutputBuffer{};
	size_t pitchOutput{};
	CUDA_CHECK(cudaMallocPitch(&tempOutputBuffer, &pitchOutput, width * 4, height));

	assert(pitchInput == pitchOutput);

	dim3 work = dim3(width, height, 1);
	dim3 numBlocks = dim3((work.x + 31) / 32, (work.y + 31) / 32, 1);
	dim3 numThreads = dim3(32, 32, 1);

	blur_kernel<<<numBlocks, numThreads>>>(tempInputBuffer, tempOutputBuffer, width, height, pitchInput, int2(params.xSize, params.ySize));
	cudaErrorPrint(cudaGetLastError());

	CUDA_CHECK(cudaMemcpy2DToArray(outImgData, 0, 0, tempOutputBuffer, pitchInput, width * 4, height, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaFree(tempInputBuffer));
	CUDA_CHECK(cudaFree(tempOutputBuffer));
}

// @TODO: sobel params
void sobelImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height, const SobelParams& params) {
	SCOPED_TIMER(__FUNCTION__);
}

} // namespace Effects