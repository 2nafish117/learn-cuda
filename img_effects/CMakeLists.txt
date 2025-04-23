cmake_minimum_required(VERSION 3.18)

set(TARGET img_effects)
add_executable(
	${TARGET} 
	main.cpp
	
	shaders.h
	shaders.cpp
	
	effects.h
	effects.cu
	)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 20)

set_target_properties(${TARGET} PROPERTIES
	WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# target_compile_options(${TARGET} PRIVATE -G)

target_link_libraries(
    ${TARGET}
    common
    stb
	glfw
	imgui

	d3d11.lib
	dxgi.lib
	d3dcompiler.lib
)