cmake_minimum_required(VERSION 3.18)

set(TARGET raytracer)

add_executable(${TARGET} 
	main.cu
)

set_target_properties(${TARGET} PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_ARCHITECTURES native
	CUDA_RUNTIME_LIBRARY Static
	CXX_STANDARD 20
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(
    ${TARGET}
    common
)