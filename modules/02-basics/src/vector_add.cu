/*****
 *  vector_add.cu — базовый пример использования сетки/блока/нитей в CUDA.
 *
 *  Задача: C = A + B для векторов длиной N.
 *
 *  Пошагово:
 *     1. Выделяем память на host и device.
 *     2. Копируем входные данные на GPU.
 *     3. Запускаем ядро vecAdd<<<grid, block>>>();
 *     4. Копируем результат обратно и проверяем.
 *
 *  grid  = ceil(N / THREADS) блоков
 *  block = THREADS (=256 нитей)
 *
 *  Сборка: cmake --build build --target vector_add
 *  Запуск: ./build/vector_add [N]
 *****/
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

constexpr int THREADS = 256; // размер CUDA-блока

/***********************
 * Ядро сложения векторов
 * idx — глобальный индекс нити (0..N-1), каждая нить обрабатывает 1 элемент.
 ***********************/
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// Утилита для проверки кода возврата CUDA API
void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int n = 1 << 20; // 1M элементов по умолчанию
    if (argc == 2) n = std::atoi(argv[1]);

    size_t bytes = n * sizeof(float);

    // === 1. Host-буферы ===
    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n);

    // === 2. Device-буферы ===
    float *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc(&d_a, bytes), "malloc a");
    checkCuda(cudaMalloc(&d_b, bytes), "malloc b");
    checkCuda(cudaMalloc(&d_c, bytes), "malloc c");

    // === 3. Копируем данные на GPU ===
    checkCuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "copy a");
    checkCuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "copy b");

    // === 4. Запускаем ядро ===
    int blocks = (n + THREADS - 1) / THREADS;
    vecAdd<<<blocks, THREADS>>>(d_a, d_b, d_c, n);
    checkCuda(cudaGetLastError(), "kernel");

    // === 5. Копируем результат на CPU ===
    checkCuda(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "copy c back");

    // === 6. Проверяем корректность ===
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(h_c[i] - 3.0f) > 1e-5f) { ok = false; break; }
    }
    printf("VectorAdd %s for %d elements\n", ok ? "PASSED" : "FAILED", n);

    // === 7. Освобождаем память ===
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
} 