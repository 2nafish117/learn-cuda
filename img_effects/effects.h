#pragma once

enum EffectsKind {
	None = 0,
	Invert,
	Blur,
	Greyscale,

	Num
};

const char* effectsKindStrings[] = {
	"None",
	"Invert",
	"Blur",
	"Greyscale",
};

namespace Effects {
	
// void invert_img(uint8_t* img_data, int width, int height, const int channels);
// void blur_img(uint8_t* img_data, int width, int height, const int channels, int blur_amt);
// void greyscale_img(uint8_t* img_data, int width, int height, const int channels, int blur_amt);

void copyImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);
void invertImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);
void greyscaleImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);
void blurImage(cudaArray_t inImgData, cudaArray_t outImgData, int width, int height);

} // namespace Effects