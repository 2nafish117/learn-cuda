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
	
void invert_img(uint8_t* img_data, int width, int height, const int channels);
void blur_img(uint8_t* img_data, int width, int height, const int channels, int blur_amt);
// void greyscale_img(uint8_t* img_data, int width, int height, const int channels, int blur_amt);

} // namespace Effects