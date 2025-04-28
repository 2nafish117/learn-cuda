#pragma once

enum EffectsKind {
	None = 0,
	Invert,
	Greyscale,
	Blur,
	Sobel,

	Num
};

static const char* const effectsKindStrings[] = {
	"None",
	"Invert",
	"Greyscale",
	"Blur",
	"Sobel (Edge Detect)",
};

namespace Effects {

void copyImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);
void invertImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);
void greyscaleImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);

// @TODO: apply a generic 2d kernel over the image, maybe port other effects to use this 
// void applyGenericKernelImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);

struct BlurParams {
	int xSize;
	int ySize;
};
void blurImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height, const BlurParams& params);

struct SobelParams {
	
};
void sobelImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height, const SobelParams& params);

} // namespace Effects