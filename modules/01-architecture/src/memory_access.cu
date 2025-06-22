#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr int THREADS = 256;

__global__ void vecAddCoalesced(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

// Некоалесцированный: читаем элементы с шагом stride
__global__ void vecAddStrided(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c,
                              int N, int stride) {
    int base = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = base * stride; // Шаг stride разрушает коалесцирование
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

void checkCuda(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

float benchmark(int N, int stride) {
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f; h_b[i] = 2.0f;
    }

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

    // Проверяем ошибки ядра
    checkCuda("kernel");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main(int argc, char** argv) {
    const int N = 1 << 24;   // ~16M элементов (64MB)
    const int stride = 32;   // некоалесцированный шаг (warp size)

    float ms_coal   = benchmark(N, 1);
    float ms_stride = benchmark(N, stride);

    printf("Vector size: %d elements\n", N);
    printf("Coalesced access time:       %.3f ms\n", ms_coal);
    printf("Strided (stride=%d) time: %.3f ms\n", stride, ms_stride);
    printf("Strided / Coalesced ratio: %.2fx\n", ms_stride / ms_coal);

    return 0;
} 