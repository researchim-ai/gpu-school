/*****
 * gemm_cublas.cu — пример использования библиотеки cuBLAS
 * для умножения матриц C = alpha * A × B + beta * C.
 *
 * Демонстрируем API:
 *   • cublasCreate / Destroy
 *   • cublasSgemm (single-precision GEMM)
 *   • Обращаем внимание, что cuBLAS использует column-major порядок (Fortran).
 *
 * Сравниваем результат с наивным CPU-GEMM.
 *****/

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cmath>

// Простая CPU-версия GEMM (row-major)
static void gemmCPU(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>&       C,
                    int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// Короткий макрос проверки ошибок cuBLAS
#define CHECK_CUBLAS(err)                                       \
    do {                                                        \
        if (err != CUBLAS_STATUS_SUCCESS) {                     \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",      \
                    static_cast<int>(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

int main(int argc, char** argv) {
    int N = 1024;                  // размер матрицы
    if (argc == 2) N = std::atoi(argv[1]);
    size_t bytes = N * N * sizeof(float);

    // Host matrices (row-major)
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N), h_ref(N * N);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
        h_B[i] = static_cast<float>((i * 3) % 100) / 100.0f;
    }

    // Device matrices in column-major (so just copy row-major, interpret accordingly)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    // cuBLAS по умолчанию работает с column-major, поэтому меняем порядок опер.
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_B, N,   // B — левая матрица в column-major → передаём как transposed
                             d_A, N,
                             &beta,
                             d_C, N));

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // CPU reference (маленькие размеры, иначе долго)
    bool validate = (N <= 512);
    double ms_cpu = 0.0;
    if (validate) {
        auto c0 = std::chrono::high_resolution_clock::now();
        gemmCPU(h_A, h_B, h_ref, N);
        auto c1 = std::chrono::high_resolution_clock::now();
        ms_cpu = std::chrono::duration<double, std::milli>(c1 - c0).count();
        bool ok = true;
        for (int i = 0; i < N * N; ++i) {
            if (fabs(h_ref[i] - h_C[i]) > 1e-2f) { ok = false; break; }
        }
        printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
    }

    printf("cuBLAS SGEMM %dx%d\n", N, N);
    printf("GPU time: %.2f ms\n", ms_gpu);
    if (validate) {
        printf("CPU time: %.2f ms\n", ms_cpu);
        printf("Speedup:  %.2fx\n", ms_cpu / ms_gpu);
    }

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
} 