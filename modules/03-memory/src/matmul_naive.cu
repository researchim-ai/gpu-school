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

// Заголовки стандартной библиотеки
#include <cstdio>      // printf / fprintf
#include <vector>      // std::vector для удобного управления памятью
#include <cuda_runtime.h> // CUDA Runtime API
#include <chrono>      // измерение времени на CPU
#include <cmath>       // fabs

// --- Параметры ядра ---
constexpr int TILE = 16; // размер блока 16×16 → 256 нитей

/***********************
 * Наивное ядро матричного умножения.
 * Каждая нить вычисляет один элемент C(row, col).
 ***********************/
__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // глобальная строка
    int col = blockIdx.x * blockDim.x + threadIdx.x; // глобальный столбец
    if (row >= N || col >= N) return;               // выход за матрицу — ничего не делаем

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];     // скалярное произведение строки и столбца
    }
    C[row * N + col] = sum;                         // сохраняем элемент
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

    size_t bytes = N * N * sizeof(float);  // общий объём данных одной матрицы N×N в байтах

    // --- Host-матрицы --- (выделяем и инициализируем)
    // h_A, h_B содержат входные данные; h_C — результат, h_ref — CPU-эталон
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N), h_ref(N * N); // host-буферы (A, B, результат C, CPU-эталон)
    for (int i = 0; i < N * N; ++i) {          // инициализация входных матриц псевдослучайн. числами
        h_A[i] = static_cast<float>(i % 100) / 100.0f;  // значение 0.00..0.99
        h_B[i] = static_cast<float>((i * 3) % 100) / 100.0f; // другое распределение
    }

    // --- Device-память --- (cudaMalloc три буфера)
    float *d_A, *d_B, *d_C;                    // device-указатели
    check(cudaMalloc(&d_A, bytes), "malloc A"); // выделяем GPU-память под A
    check(cudaMalloc(&d_B, bytes), "malloc B"); // выделяем GPU-память под B
    check(cudaMalloc(&d_C, bytes), "malloc C"); // выделяем GPU-память под C

    // Копируем входные матрицы на устройство (H2D)
    check(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice), "copy A"); // H→D копия A
    check(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice), "copy B"); // H→D копия B

    // --- Конфигурация сетки ---
    // Блок TILE×TILE, grid покрывает всю матрицу (ceil)
    dim3 block(TILE, TILE);                    // блок 16×16 нитей
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE); // grid покрывает матрицу

    // Создаём cudaEvent'ы для измерения времени
    cudaEvent_t start, stop;                   // события для замера времени на GPU
    cudaEventCreate(&start);                   // создаём событие начала
    cudaEventCreate(&stop);                    // создаём событие конца

    cudaEventRecord(start);                    // отметка T0 (до запуска ядра)
    matmulNaive<<<grid, block>>>(d_A, d_B, d_C, N); // запуск ядра умножения матриц
    check(cudaGetLastError(), "kernel");      // проверяем, что launch прошёл без ошибок
    cudaEventRecord(stop);                     // отметка T1 (сразу после запуска)

    cudaEventSynchronize(stop);             // Дожидаемся завершения ядра
    float ms = 0.f;                            // здесь будет время в мс
    cudaEventElapsedTime(&ms, start, stop);    // Δt = T1 - T0

    // --- Копируем результат на CPU (D2H) ---
    check(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "copy C back"); // D→H копия результата

    // --- CPU reference (для маленьких N, иначе слишком долго) ---
    auto t0 = std::chrono::high_resolution_clock::now();
    matmulCPU(h_A, h_B, h_ref, N);             // CPU-версия GEMM для проверки
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count(); // время CPU

    // --- Проверка корректности (abs-погрешность 1e-3) ---
    bool ok = true;
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C[i] - h_ref[i]) > 1e-3f) { ok = false; break; } // сравнение с допуском 1e-3
    }

    printf("Matrix %dx%d\n", N, N);           // вывод размера
    printf("GPU naive matmul:  %.2f ms\n", ms);
    printf("CPU reference:     %.2f ms\n", ms_cpu);
    printf("Speedup:           %.2fx\n", ms_cpu / ms);
    printf("Validation:        %s\n", ok ? "PASSED" : "FAILED"); // корректность

    // --- Очистка ресурсов девайса ---
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); // освобождаем device-память
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
} 