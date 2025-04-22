cmake_minimum_required(VERSION 3.15)

project(clog
	DESCRIPTION "c logging library" 
)

add_library(clog
	clog/log.c
)

target_include_directories(clog
	PUBLIC .
)
