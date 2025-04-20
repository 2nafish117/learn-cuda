cmake_minimum_required(VERSION 3.10)

project(learn-cuda LANGUAGES CUDA CXX)

set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

add_subdirectory(hello_world)

add_subdirectory(raytracer)