/*****
 * stream_overlap.cu — демонстрация перекрытия (overlap) передачи данных
 * и вычислений с помощью нескольких CUDA Streams.
 *
 * Сценарий:
 *   • Массивы разбиваются на чанки sizePerChunk.
 *   • Для каждого чанка создаётся собственный stream.
 *   • Сначала асинхронно копируем A[i], B[i] Host→Device.
 *   • Затем запускаем ядро сложения в том же stream.
 *   • Затем асинхронно копируем результат Device→Host.
 *
 *  При достаточном количестве чанков мы наблюдаем перекрытие копирования
 *  и вычислений → общее время < (копирование + вычисление).
 *
 * Сборка: cmake --build build --target stream_overlap
 * Запуск: ./build/stream_overlap  (по умолчанию 64 MB данных)
 *****/

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

constexpr int THREADS = 256;
constexpr int CHUNKS  = 4;  // количество стримов/чанков

__global__ void saxpy(float a, const float* x, float* y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) y[idx] = a * x[idx] + y[idx];
}

static inline void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    size_t totalBytes = 64 * (1 << 20); // 64 MB по умолчанию
    if (argc == 2) totalBytes = std::atoi(argv[1]) * (1 << 20);

    size_t totalElems = totalBytes / sizeof(float);
    size_t elemsPerChunk  = (totalElems + CHUNKS - 1) / CHUNKS;
    size_t bytesPerChunk  = elemsPerChunk * sizeof(float);

    // --- 1. Выделяем pinned host memory (ускоряет асинхр. копирование) ---
    float *h_x, *h_y;
    check(cudaHostAlloc(&h_x, totalBytes, cudaHostAllocDefault), "host alloc x");
    check(cudaHostAlloc(&h_y, totalBytes, cudaHostAllocDefault), "host alloc y");
    for (size_t i = 0; i < totalElems; ++i) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    // --- 2. Device buffers (один большой, будем адресовать смещением) ---
    float *d_x, *d_y;
    check(cudaMalloc(&d_x, totalBytes), "malloc dx");
    check(cudaMalloc(&d_y, totalBytes), "malloc dy");

    // --- 3. Создаём стримы ---
    cudaStream_t streams[CHUNKS];
    for (int i = 0; i < CHUNKS; ++i) cudaStreamCreate(&streams[i]);

    auto t0 = std::chrono::high_resolution_clock::now();

    // --- 4. Обработка чанков ---
    for (int c = 0; c < CHUNKS; ++c) {
        size_t offsetElem = c * elemsPerChunk;
        size_t offsetByte = c * bytesPerChunk;
        size_t thisElems  = (c == CHUNKS - 1) ? (totalElems - offsetElem) : elemsPerChunk;
        size_t thisBytes  = thisElems * sizeof(float);

        cudaStream_t s = streams[c];

        // H2D копия X и Y
        check(cudaMemcpyAsync(d_x + offsetElem, h_x + offsetElem, thisBytes,
                              cudaMemcpyHostToDevice, s), "H2D x");
        check(cudaMemcpyAsync(d_y + offsetElem, h_y + offsetElem, thisBytes,
                              cudaMemcpyHostToDevice, s), "H2D y");

        // Конфигурация ядра
        int blocks = (thisElems + THREADS - 1) / THREADS;
        saxpy<<<blocks, THREADS, 0, s>>>(2.0f, d_x + offsetElem, d_y + offsetElem, thisElems);
        check(cudaGetLastError(), "kernel");

        // D2H копия результата
        check(cudaMemcpyAsync(h_y + offsetElem, d_y + offsetElem, thisBytes,
                              cudaMemcpyDeviceToHost, s), "D2H y");
    }

    // --- 5. Синхронизируемся со всеми стримами ---
    check(cudaDeviceSynchronize(), "device sync");

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- 6. Проверяем корректность (берём 5 случайных точек) ---
    bool ok = true;
    for (int i = 0; i < 5; ++i) {
        size_t idx = (i * 997) % totalElems;
        if (fabs(h_y[idx] - (2.0f * 1.0f + 2.0f)) > 1e-5f) { ok = false; break; }
    }

    printf("Total data: %.2f MB, chunks: %d, time: %.2f ms, %.2f GB/s, %s\n",
           totalBytes / 1e6, CHUNKS, ms_total, (totalBytes * 3) / 1e6 / ms_total, ok ? "OK" : "FAIL");

    for (auto& s : streams) cudaStreamDestroy(s);
    cudaFree(d_x); cudaFree(d_y);
    cudaFreeHost(h_x); cudaFreeHost(h_y);
    return ok ? 0 : 1;
} 