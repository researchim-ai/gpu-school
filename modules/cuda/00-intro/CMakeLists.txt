cmake_minimum_required(VERSION 3.20)

project(hello_gpu LANGUAGES CUDA)

add_executable(hello_gpu src/hello_gpu.cu)

# Включаем раздельную компиляцию для больших проектов
set_target_properties(hello_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(hello_gpu PUBLIC cxx_std_17) 