add_executable(vector_add src/vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(vector_add PUBLIC cxx_std_17) 