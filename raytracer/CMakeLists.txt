
set(TARGET raytracer)

add_executable(${TARGET} 
	main.cu
)

target_compile_options(${TARGET} PRIVATE -G)
