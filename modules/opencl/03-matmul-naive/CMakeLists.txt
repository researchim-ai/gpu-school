cmake_minimum_required(VERSION 3.20)

project(matmul_naive_cl LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

add_executable(matmul_naive_cl src/matmul_naive_cl.cpp)

target_link_libraries(matmul_naive_cl PRIVATE OpenCL::OpenCL)

target_compile_features(matmul_naive_cl PUBLIC cxx_std_17) 