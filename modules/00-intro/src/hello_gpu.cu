/*****
 *  Hello GPU — самый первый пример CUDA.
 *
 *  Цель:
 *    • проверить, что окружение и драйверы работают;
 *    • показать базовый синтаксис __global__-ядра;
 *    • напечатать уникальный идентификатор каждой нити.
 *
 *  Как собрать:
 *        cmake --build build --target hello_gpu   # после конфигурирования CMake
 *
 *  Как работает:
 *    1. CPU (хост) вызывает hello_kernel<<<1, 32>>>();
 *       Это значит: «создай grid из 1 блока, в блоке 32 нитей».
 *    2. Каждая нить вычисляет свой глобальный индекс и печатает сообщение.
 *    3. После завершения ядра вызываем cudaDeviceSynchronize() —
 *       дожидаемся конца выполнения, чтобы printf успел записать вывод.
 *    4. Проверяем cudaGetLastError() на случай проблем во время запуска.
 *****/
#include <cstdio>
#include <cuda_runtime.h>

// __global__ указывает, что функция выполняется на устройстве (GPU)
// и может быть вызвана только с хоста.
__global__ void hello_kernel() {
    // threadIdx.x — локальный идентификатор нити в блоке
    // blockIdx.x   — идентификатор блока в grid
    // blockDim.x   — размер блока (число нитей)
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from GPU thread %d\n", global_id);
}

int main() {
    // 1 блок × 32 нити  → всего 32 нити
    hello_kernel<<<1, 32>>>();

    // Синхронизируемся: ждём завершения GPU, иначе программа может завершиться раньше
    cudaDeviceSynchronize();

    // Проверяем возможные ошибки запуска ядра
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} 