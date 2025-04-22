cmake_minimum_required(VERSION 3.15)

project(cgltf
	DESCRIPTION "cgltf header libraries" 
)

add_library(cgltf
	cgltf/cgltf.h
	cgltf/cgltf_write.h
	cgltf.cpp
)

target_include_directories(cgltf
	PUBLIC .
)
