cmake_minimum_required(VERSION 3.20)

project(reduction_atomic_cl LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

add_executable(reduction_atomic_cl src/reduction_atomic_cl.cpp)

target_link_libraries(reduction_atomic_cl PRIVATE OpenCL::OpenCL)

target_compile_features(reduction_atomic_cl PUBLIC cxx_std_17) 