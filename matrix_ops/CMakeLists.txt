cmake_minimum_required(VERSION 3.18)

set(TARGET matrix_ops)
add_executable(${TARGET} main.cu)

set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD 20)

# target_compile_options(${TARGET} PRIVATE -G)

target_link_libraries(
    ${TARGET}
    common
)