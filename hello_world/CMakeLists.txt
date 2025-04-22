cmake_minimum_required(VERSION 3.18)

set(TARGET hello_world)
add_executable(${TARGET} main.cu)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)
# target_compile_options(${TARGET} PRIVATE -G)
