cmake_minimum_required(VERSION 3.15)

project(stb
	DESCRIPTION "stb header libraries" 
)

add_library(stb
	stb.cpp
)

target_include_directories(stb
	PUBLIC .
)
