cmake_minimum_required(VERSION 3.20)

project(gpu_school LANGUAGES C CXX CUDA)

# Размещаем все исполняемые файлы прямо в корневой директории сборки
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Добавляем модули курса
add_subdirectory(modules/00-intro)
add_subdirectory(modules/01-architecture)
add_subdirectory(modules/02-basics)
add_subdirectory(modules/03-memory)
add_subdirectory(modules/04-performance)
add_subdirectory(modules/06-advanced)
add_subdirectory(modules/07-libraries)
add_subdirectory(modules/08-profiling)

# Включаем тесты, чтобы примеры можно было запускать через `ctest`
enable_testing()

add_test(NAME hello_gpu COMMAND hello_gpu)
add_test(NAME memory_access COMMAND memory_access)
add_test(NAME vector_add COMMAND vector_add)
add_test(NAME matmul_naive COMMAND matmul_naive)
add_test(NAME matmul_tiled COMMAND matmul_tiled)
add_test(NAME graph_vector_add COMMAND graph_vector_add)
add_test(NAME gemm_cublas COMMAND gemm_cublas) 