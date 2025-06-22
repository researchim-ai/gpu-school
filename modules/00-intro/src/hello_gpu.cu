#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\n", threadIdx.x + blockIdx.x * blockDim.x);
}

int main() {
    // Запускаем 1 блок из 32 потоков
    hello_kernel<<<1, 32>>>();

    // Ожидаем завершения GPU
    cudaDeviceSynchronize();

    // Проверяем, не возникло ли ошибок при запуске ядра
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
} 