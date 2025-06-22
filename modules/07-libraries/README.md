# Модуль 7 — CUDA Libraries (cuBLAS)

Библиотека cuBLAS предоставляет высокооптимизированные BLAS-рутины (GEMM, GEMV, GEMM-STRIDED-BATCH и т. д.), использующие современные возможности GPU (Tensor Cores).

## Пример `gemm_cublas.cu`

– Однократный вызов `cublasSgemm` для матриц `N×N`.  
– Показана разница row-major vs column-major.  
– Сравнение с CPU-референсом для N ≤ 512.

### Сборка и запуск

```bash
cmake --build build --target gemm_cublas -j$(nproc)
./build/gemm_cublas          # 1024×1024 по умолчанию
./build/gemm_cublas 4096     # 4096×4096 (долго на CPU)
```

### Теория

| BLAS уровень | Операция | cuBLAS функция |
|--------------|----------|----------------|
| Level 1 | `y = αx + y` (AXPY) | `cublasSaxpy` |
| Level 2 | `y = αA·x + βy`    | `cublasSgemv` |
| Level 3 | `C = αA·B + βC`    | `cublasSgemm` |

Память в cuBLAS трактуется **column-major**. Если ваши данные row-major (C-style), можно:
1. Транспонировать матрицы при копировании.  
2. Поменять порядок аргументов (см. исходник примера).  
3. Использовать флаг `CUBLAS_OP_T`.

### Задания
1. Измерьте производительность в FLOPS: `2*N^3 / time`.  
2. Включите Tensor Cores: `cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)` и сравните.  
3. Реализуйте `cublasSgemmStridedBatched` для мини-батча 32 матриц. 