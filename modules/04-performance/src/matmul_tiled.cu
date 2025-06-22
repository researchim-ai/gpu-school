/*****
 *  matmul_tiled.cu — оптимизированное матричное умножение (GEMM)
 *  с использованием тайлов и shared memory.
 *
 *  Улучшения по сравнению с наивной версией (Module 3):
 *    • Каждая нить читает элементы A и B в shared memory.
 *    • Блок 16×16 нитей вычисляет подматрицу 16×16.
 *    • Сокращаем обращения к глобальной памяти.
 *
 *  Сборка:  cmake --build build --target matmul_tiled
 *  Запуск:  ./build/matmul_tiled [N]
 *****/

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

constexpr int TILE = 16; // размер тайла и блока

__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N) {
    // Координаты нити
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Тайловые буферы в shared memory
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    // Проходим по тайлам матриц A и B
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        // Загружаем элементы в shared memory (проверяем границы)
        int tiledRow = row;
        int tiledCol = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (tiledRow < N && tiledCol < N) ? A[tiledRow * N + tiledCol] : 0.0f;

        tiledRow = t * TILE + threadIdx.y;
        tiledCol = col;
        Bs[threadIdx.y][threadIdx.x] = (tiledRow < N && tiledCol < N) ? B[tiledRow * N + tiledCol] : 0.0f;

        __syncthreads();

        // Вычисляем частичную сумму
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Записываем результат
    if (row < N && col < N)
        C[row * N + col] = sum;
}

static inline void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void matmulCPU(const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>& C,
               int N) {
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            float s = 0.f;
            for (int k = 0; k < N; ++k)
                s += A[r * N + k] * B[k * N + c];
            C[r * N + c] = s;
        }
}

int main(int argc, char** argv) {
    int N = 512;
    if (argc == 2) N = std::atoi(argv[1]);
    size_t bytes = N * N * sizeof(float);

    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N), h_ref(N * N);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
        h_B[i] = static_cast<float>((i * 3) % 100) / 100.0f;
    }

    float *d_A, *d_B, *d_C;
    check(cudaMalloc(&d_A, bytes), "malloc A");
    check(cudaMalloc(&d_B, bytes), "malloc B");
    check(cudaMalloc(&d_C, bytes), "malloc C");

    check(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice), "copy A");
    check(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice), "copy B");

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    check(cudaGetLastError(), "kernel");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_gpu = 0.f;
    cudaEventElapsedTime(&ms_gpu, start, stop);

    check(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "copy back");

    // CPU ref for validation (optional, for small N)
    bool validate = (N <= 512);
    double ms_cpu = 0.0;
    if (validate) {
        auto t0 = std::chrono::high_resolution_clock::now();
        matmulCPU(h_A, h_B, h_ref, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bool ok = true;
        for (int i = 0; i < N * N; ++i) {
            if (fabs(h_C[i] - h_ref[i]) > 1e-3f) { ok = false; break; }
        }
        printf("Validation: %s\n", ok ? "PASSED" : "FAILED");
    }

    printf("Matrix %dx%d\n", N, N);
    printf("GPU tiled matmul:  %.2f ms\n", ms_gpu);
    if (validate) {
        printf("CPU reference:     %.2f ms\n", ms_cpu);
        printf("Speedup:           %.2fx\n", ms_cpu / ms_gpu);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
} 