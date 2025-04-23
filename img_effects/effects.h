#pragma once

namespace Effects {
	
void invert_img(uint8_t* img_data, int width, int height, const int channels);
void blur_img(uint8_t* img_data, int width, int height, const int channels, int blur_amt);
// void greyscale_img(uint8_t* img_data, int width, int height, const int channels, int blur_amt);

} // namespace Effects