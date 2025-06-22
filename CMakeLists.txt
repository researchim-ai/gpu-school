cmake_minimum_required(VERSION 3.20)

project(gpu_school LANGUAGES C CXX CUDA)

# Добавляем модули курса
add_subdirectory(modules/00-intro)
add_subdirectory(modules/01-architecture)
add_subdirectory(modules/02-basics)

# Размещаем все исполняемые файлы прямо в корневой директории сборки
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Включаем тесты, чтобы примеры можно было запускать через `ctest`
enable_testing()

add_test(NAME hello_gpu COMMAND hello_gpu)
add_test(NAME memory_access COMMAND memory_access)
add_test(NAME vector_add COMMAND vector_add) 