/*****
 *  vector_add.cu — подробное объяснение каждой части программы.
 *****/
// Заголовки C/C++
#include <cstdio>      // printf, fprintf
#include <vector>      // std::vector (упрощает управление памятью на CPU)
#include <cmath>       // fabs

// Заголовок Runtime API CUDA
#include <cuda_runtime.h>

// ---------------- Параметры ядра ----------------
constexpr int THREADS = 256; // 256 нитей на блок — типичное значение

/********************************************************
 * Глобальная функция (ядро) vecAdd
 * Каждый поток обрабатывает один элемент векторов A, B.
 *
 * Параметры:
 *   a, b — указатели на device-память (входные вектора)
 *   c     — выходной вектор в device-памяти
 *   n     — количество элементов
 ********************************************************/
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    // Вычисляем глобальный индекс текущей нити
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)                     // проверяем границу, чтобы не выйти за массив
        c[idx] = a[idx] + b[idx];    // выполняем собственно сложение
}

/********************************************************
 * checkCuda — утилитарная функция проверки ошибок.
 *   err — код, возвращаемый CUDA API
 *   msg — строка, выводимая при ошибке
 ********************************************************/
static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    // ---- 1. Задаём размер вектора ----
    int n = 1 << 20;                 // 1M элементов по умолчанию
    if (argc == 2) n = std::atoi(argv[1]); // можно переопределить из CLI

    size_t bytes = n * sizeof(float); // сколько байт нужно на массив

    // ---- 2. Выделяем host-память через std::vector ----
    std::vector<float> h_a(n, 1.0f); // заполнить 1.0
    std::vector<float> h_b(n, 2.0f); // заполнить 2.0
    std::vector<float> h_c(n);       // итоговый вектор

    // ---- 3. Выделяем device-память ----
    float *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc(&d_a, bytes), "cudaMalloc A");
    checkCuda(cudaMalloc(&d_b, bytes), "cudaMalloc B");
    checkCuda(cudaMalloc(&d_c, bytes), "cudaMalloc C");

    // ---- 4. Копируем данные с CPU на GPU ----
    checkCuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "H2D B");

    // ---- 5. Запускаем ядро ----
    int blocks = (n + THREADS - 1) / THREADS; // округление вверх
    vecAdd<<<blocks, THREADS>>>(d_a, d_b, d_c, n);
    checkCuda(cudaGetLastError(), "kernel launch"); // проверяем ошибки запуска

    // ---- 6. Копируем результат обратно ----
    checkCuda(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "D2H C");

    // ---- 7. Проверка корректности ----
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(h_c[i] - 3.0f) > 1e-5f) { ok = false; break; }
    }
    printf("VectorAdd %s for %d elements\n", ok ? "PASSED" : "FAILED", n);

    // ---- 8. Освобождаем ресурсы ----
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
} 