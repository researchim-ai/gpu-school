/*****
 *  memory_access.cu — демонстрация коалесцированного и не-коалесцированного
 *  (strided) доступа к глобальной памяти GPU.
 *
 *  • vecAddCoalesced  — каждая нить читает соседний элемент -> обращения сливаются
 *  • vecAddStrided    — каждая нить читает через stride (32) -> обращения разбиваются
 *
 *  Программа измеряет время выполнения обоих ядер для большого вектора и выводит
 *  коэффициент замедления.
 *
 *  Сборка (после конфигурации CMake):
 *        cmake --build build --target memory_access
 *  Запуск:
 *        ./build/memory_access
 *****/
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int THREADS = 256;   // размер блока (размер warp = 32)

/***************
 * Ядро с коалесцированным доступом.
 * Память читается/записывается последовательно: a[idx], b[idx].
 * Контроллер памяти GPU может слить обращения нитей варпа в один request.
 ***************/
__global__ void vecAddCoalesced(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

/***************
 * Ядро с «дырявым» (strided) доступом.
 * Каждая нить обращается к элементу через stride, разрушая последовательность.
 * В результате контроллер делает N/stride транзакций вместо N/32 (примерно).
 ***************/
__global__ void vecAddStrided(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c,
                              int N, int stride) {
    int base = blockDim.x * blockIdx.x + threadIdx.x;
    int idx  = base * stride; // через stride
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Простая обёртка для проверки последней CUDA-ошибки
void checkCuda(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/***************
 * benchmark() — выделяет память, заполняет её, запускает ядро и измеряет время
 * с помощью cudaEventElapsedTime (миллисекунды).
 ***************/
float benchmark(int N, int stride) {
    size_t bytes = N * sizeof(float);

    // Выделяем host-буферы и инициализируем
    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f; h_b[i] = 2.0f;
    }

    // Выделяем device-буферы
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS - 1) / THREADS;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (stride == 1) {
        vecAddCoalesced<<<blocks, THREADS>>>(d_a, d_b, d_c, N);
    } else {
        vecAddStrided<<<blocks, THREADS>>>(d_a, d_b, d_c, N, stride);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);

    checkCuda("kernel");

    // Освобождаем ресурсы
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a);     free(h_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    const int N      = 1 << 24; // ~16 млн элементов (64 МБ)
    const int stride = 32;      // типичный размер warp

    float ms_coal   = benchmark(N, 1);
    float ms_stride = benchmark(N, stride);

    printf("Vector size: %d elements\n", N);
    printf("Coalesced access time:       %.3f ms\n", ms_coal);
    printf("Strided (stride=%d) time: %.3f ms\n", stride, ms_stride);
    printf("Strided / Coalesced ratio: %.2fx\n", ms_stride / ms_coal);

    return 0;
} 