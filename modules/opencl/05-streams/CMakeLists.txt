cmake_minimum_required(VERSION 3.20)

project(stream_overlap_cl LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

add_executable(stream_overlap_cl src/stream_overlap_cl.cpp)

target_link_libraries(stream_overlap_cl PRIVATE OpenCL::OpenCL)

target_compile_features(stream_overlap_cl PUBLIC cxx_std_17) 