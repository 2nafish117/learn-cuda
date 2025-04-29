# im learning cuda

## building and running

1. make sure you have cuda installed. 

2. clone the repo
```
git clone https://github.com/2nafish117/learn-cuda
```

1. from the root of the repo run
```
mkdir build && cd build
cmake ..
```

1. run the generated visual studio solution in the build directory

## TODO

1. effects
   1. unify application of kernels onto images
   2. canny edge detection filter
   3. sobel edge detection filter
   4. gaussian blur with params
2. raytracer/pathtracer
   1. draw sphere
   2. draw plane
   3. simple shading
   4. make scene struct, primitives list
   5. https://raytracing.github.io/books/RayTracingInOneWeekend.html

## writeup on
1. separating cuda and cpp code