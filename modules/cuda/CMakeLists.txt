cmake_minimum_required(VERSION 3.20)

# Агрегируем все подмодули CUDA
add_subdirectory(00-intro)
add_subdirectory(01-memory-access)
add_subdirectory(02-vector-add)
add_subdirectory(03-matmul-naive)
add_subdirectory(04-matmul-tiled)
add_subdirectory(05-streams)
add_subdirectory(06-graph-api)
add_subdirectory(07-cublas)
add_subdirectory(08-profiling)