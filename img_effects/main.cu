#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <chrono>
#include <random>
#include <cassert>
#include <filesystem>
#include <direct.h>

#include <stb/stb_image.h>
#include <common/common.ch>

int main() {
	int width, height, channels;
	
	// spdlog::info("{}", std::filesystem::current_path().is_absolute());
	printf("%ls", std::filesystem::current_path().c_str());

	// Define a buffer 
    const size_t size = 1024; 
    // Allocate a character array to store the directory path
    char buffer[size];
    
    // Call _getcwd to get the current working directory and store it in buffer
    if (getcwd(buffer, size) != NULL) {
        // print the current working directory
        printf("Current working directory: %s", buffer);
    } 
    else {
        // If _getcwd returns NULL, print an error message
        std::cerr << "Error getting current working directory" << std::endl;
    }

	uint8_t* data = stbi_load("../assets/benjamin.jpg", &width, &height, &channels, 3);

	stbi_image_free(data);

	return 0;
}