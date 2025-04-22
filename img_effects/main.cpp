#include <iostream>
#include <chrono>
#include <random>
#include <cassert>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

#include "effects.h"

int main() {
	int width, height, channels;
	
	// cwd of exe D:\Dev\learn-cuda\build\img_effects\Debug
	std::string assets_path = "../../../assets/";
	std::string img_path = assets_path + "/benjamin.jpg";
	std::string out_img_path = assets_path + "/blurred_benjamin.jpg";

	const int desired_channels = 3;
	uint8_t* data = stbi_load(img_path.c_str(), &width, &height, &channels, desired_channels);

	Effects::blur_img(data, width, height, desired_channels);
	stbi_write_jpg(out_img_path.c_str(), width, height, desired_channels, data, 100);

	stbi_image_free(data);

	return 0;
}