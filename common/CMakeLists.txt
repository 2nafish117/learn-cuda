cmake_minimum_required(VERSION 3.18)

project(common
	DESCRIPTION "common stuff" 
)

set(TARGET common)

add_library(${TARGET}
	common/common.h
	common/common.cpp
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

# target_link_libraries(
#     ${TARGET}
# )

target_include_directories(${TARGET}
	PUBLIC .
)
