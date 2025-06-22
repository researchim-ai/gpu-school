/*****
 *  matmul_naive.cu — демонстрация управления памятью и наивного GEMM
 *  (C = A × B) без оптимизаций.
 *
 *  Цели примера:
 *    • Показать выделение device-памяти cudaMalloc и копирование cudaMemcpy.
 *    • Использовать глобальную память без shared memory → базовая производительность.
 *    • Сравнить время CPU vs GPU.
 *
 *  Сборка:  cmake --build build --target matmul_naive
 *  Запуск:  ./build/matmul_naive [N]
 *           где N — размер квадратной матрицы (по умолчанию 512)
 *****/

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// --- Параметры ядра ---
constexpr int TILE = 16; // размер блока 16×16 → 256 нитей

/***********************
 * Наивное ядро матричного умножения.
 * Каждая нить вычисляет один элемент C(row, col).
 ***********************/
__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// --- Утилита проверки ошибок ---
static inline void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// --- Простейшая CPU-референсная реализация ---
void matmulCPU(const std::vector<float>& A,
               const std::vector<float>& B,
               std::vector<float>&       C,
               int N) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[row * N + k] * B[k * N + col];
            C[row * N + col] = sum;
        }
    }
}

int main(int argc, char** argv) {
    int N = 512; // размер по умолчанию (512×512)
    if (argc == 2) N = std::atoi(argv[1]);

    size_t bytes = N * N * sizeof(float);

    // --- Host-матрицы ---
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N), h_ref(N * N);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
        h_B[i] = static_cast<float>((i * 3) % 100) / 100.0f;
    }

    // --- Device-память ---
    float *d_A, *d_B, *d_C;
    check(cudaMalloc(&d_A, bytes), "malloc A");
    check(cudaMalloc(&d_B, bytes), "malloc B");
    check(cudaMalloc(&d_C, bytes), "malloc C");

    check(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice), "copy A");
    check(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice), "copy B");

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // --- GPU запуск + замер времени cudaEvent ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    check(cudaGetLastError(), "kernel");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    // --- Копируем результат на CPU ---
    check(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "copy C back");

    // --- CPU reference для проверки и сравнения скорости ---
    auto t0 = std::chrono::high_resolution_clock::now();
    matmulCPU(h_A, h_B, h_ref, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- Проверка корректности ---
    bool ok = true;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C[i] - h_ref[i]) > 1e-3f) { ok = false; break; }
    }

    printf("Matrix %dx%d\n", N, N);
    printf("GPU naive matmul:  %.2f ms\n", ms);
    printf("CPU reference:     %.2f ms\n", ms_cpu);
    printf("Speedup:           %.2fx\n", ms_cpu / ms);
    printf("Validation:        %s\n", ok ? "PASSED" : "FAILED");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
} 