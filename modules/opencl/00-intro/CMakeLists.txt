cmake_minimum_required(VERSION 3.20)

project(hello_opencl LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

add_executable(hello_opencl src/hello_opencl.cpp)

target_link_libraries(hello_opencl PRIVATE OpenCL::OpenCL)

target_compile_features(hello_opencl PUBLIC cxx_std_17) 