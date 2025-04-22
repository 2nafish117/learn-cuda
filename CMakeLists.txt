cmake_minimum_required(VERSION 3.18)

project(learn-cuda LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Debug)

add_subdirectory(hello_world)
add_subdirectory(matrix_ops)
add_subdirectory(raytracer)
add_subdirectory(blurs)

add_subdirectory(vendor)