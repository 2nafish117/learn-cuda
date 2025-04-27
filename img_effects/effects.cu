#include <common/common.h>

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
void invert_kernel2(
	cudaArray_t inImgData, cudaArray_t outImgData, 
	const int width, const int height) 
{
	const size_t pixel_index = blockDim.x * blockIdx.x + threadIdx.x;

	// tex1D<float>(inImgData, 1, 2);
	// for(int i = 0; i < 4; ++i) {
	// 	const size_t comp_index = 4 * pixel_index + i;
	// 	uint8_t comp_val = in_img_data[comp_index];
	// 	out_img_data[comp_index] = 255 - comp_val;
	// }
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
	uchar4* img_data,
	uchar4* out_img_data,
	int width, int height)
{
	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
	float red = 	(float) img_data[index].x / 255.0f;
	float green = 	(float) img_data[index].y / 255.0f;
	float blue = 	(float) img_data[index].z / 255.0f;

	float greyscale = 0.299f * red + 0.587f * green + 0.114f * blue;
	out_img_data[index] = uchar4(greyscale * 255.0f);
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

void copyImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height) {
	CUDA_CHECK(cudaMemcpy2DArrayToArray(outImgData, 0, 0, inImgData, 0, 0, width * 4, height));
}

void invertImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height) {
	SCOPED_TIMER(__FUNCTION__);

	// @TODO: cuda cannot directly modify a d3d texture via mapping and using the cudaArray, we need to memcpy to the array from
	// a separate buffer that we wrote to
	// cudaMallocPitch()

	// we know the image has 4 components, rgba
	const size_t size = width * height * 4;

	dim3 work = dim3(width * height, 1, 1);
	dim3 numBlocks = dim3((work.x + 1023) / 1024, 1, 1);
	dim3 numThreads = dim3(1024, 1, 1);

	invert_kernel2<<<numBlocks, numThreads>>>(inImgData, outImgData, width, height);
	cudaErrorPrint(cudaGetLastError());
}

void greyscaleImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height) {
	// @TODO:
}

void blurImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height) {
	// @TODO:
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

// void greyscale_img(
// 	uint8_t* img_data, 
// 	uint8_t* out_img_data, 
// 	int width, int height)
// {
// 	SCOPED_TIMER(__FUNCTION__);
	
// 	const size_t size = width * height;

// 	uchar4* device_img_data = nullptr;
// 	CUDA_CHECK(cudaMalloc(&device_img_data, size));
// 	CUDA_CHECK(cudaMemcpy(device_img_data, img_data, size, cudaMemcpyKind::cudaMemcpyHostToDevice));
	
// 	dim3 work = dim3(width, height, 1);
// 	dim3 numBlocks = dim3((width + 31) / 32, (height + 31) / 32, 1);
// 	dim3 numThreads = dim3(32, 32, 1);
// 	greyscale_kernel<<<1, 1>>>(device_img_data, device_img_data, width, height);
	
// 	CUDA_CHECK(cudaMemcpy(img_data, device_img_data, size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
// 	CUDA_CHECK(cudaFree(device_img_data));
// }

} // namespace Effects