
add_executable(hello_world main.cu)

target_compile_options(hello_world PRIVATE -G)

# Set any additional include directories or libraries if needed
# target_include_directories(executable1 PRIVATE /path/to/includes)

# Link any libraries if necessary
# target_link_libraries(executable1 PRIVATE my_library)