cmake_minimum_required(VERSION 3.20)

project(saxpy_cl LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

add_executable(saxpy_basic_cl src/saxpy_cl.cpp)

target_link_libraries(saxpy_basic_cl PRIVATE OpenCL::OpenCL)

target_compile_features(saxpy_basic_cl PUBLIC cxx_std_17) 