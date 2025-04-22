cmake_minimum_required(VERSION 3.15)

project(vendor
	DESCRIPTION "vendor libraries" 
)

# stb stuff
add_subdirectory(stb)
set_target_properties(stb PROPERTIES FOLDER "Vendor")

# log.c stuff
add_subdirectory(clog)
set_target_properties(clog PROPERTIES FOLDER "Vendor")