cmake_minimum_required(VERSION 3.20)

project(gpu_school LANGUAGES C CXX CUDA)

# Размещаем все исполняемые файлы прямо в корневой директории сборки
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Добавляем модули курса
add_subdirectory(modules/cuda)
add_subdirectory(modules/opencl/00-intro)
add_subdirectory(modules/opencl/02-saxpy)
add_subdirectory(modules/opencl/03-matmul-naive)
add_subdirectory(modules/opencl/04-matmul-tiled)
add_subdirectory(modules/opencl/05-streams)
add_subdirectory(modules/opencl/06-reduction-atomic)
add_subdirectory(modules/opencl/07-clblast)

# Включаем тесты, чтобы примеры можно было запускать через `ctest`
enable_testing()

add_test(NAME hello_gpu COMMAND hello_gpu)
add_test(NAME memory_access COMMAND memory_access)
add_test(NAME vector_add COMMAND vector_add)
add_test(NAME matmul_naive COMMAND matmul_naive)
add_test(NAME matmul_tiled COMMAND matmul_tiled)
add_test(NAME graph_vector_add COMMAND graph_vector_add)
add_test(NAME gemm_cublas COMMAND gemm_cublas)
add_test(NAME hello_opencl COMMAND hello_opencl)
add_test(NAME saxpy_basic_cl COMMAND saxpy_basic_cl)
add_test(NAME matmul_naive_cl COMMAND matmul_naive_cl 128)
add_test(NAME matmul_tiled_cl COMMAND matmul_tiled_cl 256)
add_test(NAME stream_overlap_cl COMMAND stream_overlap_cl 32)
add_test(NAME reduction_atomic_cl COMMAND reduction_atomic_cl 1048576)
if(TARGET gemm_clblast)
  add_test(NAME gemm_clblast COMMAND gemm_clblast 512)
endif() 