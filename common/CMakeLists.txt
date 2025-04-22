cmake_minimum_required(VERSION 3.18)

project(common
	DESCRIPTION "common stuff" 
)

set(TARGET common)

add_library(${TARGET}
	common/common.cu
)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 20)
# target_compile_options(${TARGET} PRIVATE -G)

target_link_libraries(
    ${TARGET}
)

target_include_directories(${TARGET}
	PUBLIC .
)
