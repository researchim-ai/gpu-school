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
#include <cstdio>              // printf на стороне CPU и внутри GPU (через device printf)
#include <cuda_runtime.h>      // Основной заголовок CUDA Runtime API

/*****
 * Каждая функция, помеченная __global__, компилируется в GPU-ядро.
 * Её может вызвать только хост-код (CPU). Возврат всегда void.
 *****/
__global__ void hello_kernel() {
    // threadIdx.x — локальный индекс внутри блока (0..blockDim.x-1)
    // blockIdx.x   — индекс блока внутри grid
    // blockDim.x   — размер блока (число нитей)
    int global_id = threadIdx.x + blockIdx.x * blockDim.x; // Вычисляем глобальный ID нити
    printf("Hello from GPU thread %d\n", global_id);        // device-printf (работает медленно, но удобен для дебага)
}

int main() {
    /*****
     * Синтаксис <<<Grid, Block>>> указывает размеры запуска ядра:
     *   • Grid  = 1 блок  (первый параметр)
     *   • Block = 32 нитей (второй параметр)
     * Всего создаётся 1 × 32 = 32 нити.
     *****/
    hello_kernel<<<1, 32>>>(); // Асинхронный вызов: CPU сразу продолжит выполнение

    cudaDeviceSynchronize();   // Блокируем CPU до завершения всех предыдущих GPU-работ

    // Проверяем, не случилось ли ошибки при запуске ядра или во время его выполнения
    cudaError_t err = cudaGetLastError();           // Возвращает код последней ошибки Runtime API / ядра
    if (err != cudaSuccess) {                       // cudaSuccess == 0 → всё хорошо
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;                        // Выходим с кодом 1 при ошибке
    }

    return EXIT_SUCCESS;        // Код 0 → успех
} 