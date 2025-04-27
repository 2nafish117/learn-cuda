#pragma once

enum EffectsKind {
	None = 0,
	Invert,
	Greyscale,
	Blur,
	Sobel,

	Num
};

const char* effectsKindStrings[] = {
	"None",
	"Invert",
	"Greyscale",
	"Blur (Box)",
	"Sobel (Edge Detect)",
};

namespace Effects {

void copyImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);
void invertImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);
void greyscaleImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);
void blurImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);
void sobelImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);

} // namespace Effects