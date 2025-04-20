cmake_minimum_required(VERSION 3.18)

set(TARGET raytracer)

add_executable(${TARGET} 
	main.cu
)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES OFF)
target_compile_options(${TARGET} PRIVATE -G)
