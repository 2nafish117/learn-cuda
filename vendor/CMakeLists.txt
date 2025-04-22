cmake_minimum_required(VERSION 3.15)

project(vendor
	DESCRIPTION "vendor libraries" 
)

# glfw stuff
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(glfw-3.4)
set_target_properties(glfw PROPERTIES FOLDER "Vendor/GLTF3")
set_target_properties(update_mappings PROPERTIES FOLDER "Vendor/GLTF3")

# spdlog stuff
# set(SPDLOG_WCHAR_SUPPORT ON)
# set(SPDLOG_WCHAR_FILENAMES ON)
# add_subdirectory(spdlog-1.14.1)
# set_target_properties(spdlog PROPERTIES FOLDER "Vendor")

# cgltf stuff
add_subdirectory(cgltf)
set_target_properties(cgltf PROPERTIES FOLDER "Vendor")

# stb stuff
add_subdirectory(stb)
set_target_properties(stb PROPERTIES FOLDER "Vendor")

# imgui stuff
add_subdirectory(imgui)
set_target_properties(imgui PROPERTIES FOLDER "Vendor")

# flags arg parser stuff
add_subdirectory(flags-1.1)
set_target_properties(flags PROPERTIES FOLDER "Vendor")