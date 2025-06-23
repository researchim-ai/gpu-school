cmake_minimum_required(VERSION 3.20)

project(matmul_tiled_cl LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

add_executable(matmul_tiled_cl src/matmul_tiled_cl.cpp)

target_link_libraries(matmul_tiled_cl PRIVATE OpenCL::OpenCL)

target_compile_features(matmul_tiled_cl PUBLIC cxx_std_17) 