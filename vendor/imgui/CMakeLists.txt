cmake_minimum_required(VERSION 3.15)

project(vendor
	DESCRIPTION "vendor libraries" 
)

add_library(imgui
    "imgui/imgui.h"
    "imgui/imgui.cpp"
    "imgui/imgui_demo.cpp"
    "imgui/imgui_widgets.cpp"
    "imgui/imgui_tables.cpp"
    "imgui/imgui_draw.cpp"
    "imgui/backends/imgui_impl_dx11.h"
    "imgui/backends/imgui_impl_dx11.cpp"
    "imgui/backends/imgui_impl_glfw.h"
    "imgui/backends/imgui_impl_glfw.cpp"
)

target_include_directories(imgui
    PUBLIC ${CMAKE_SOURCE_DIR}/vendor/imgui/imgui/
)

target_link_libraries(imgui
    glfw
)
