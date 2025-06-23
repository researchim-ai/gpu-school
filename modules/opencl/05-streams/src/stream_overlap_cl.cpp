/*
 * stream_overlap_cl.cpp — демонстрация перекрытия (overlap) копирования и вычислений в OpenCL.
 * Использует несколько независимых очередей команд, каждая обрабатывает свой «чанк» данных.
 * Алгоритм: SAXPY (y = a * x + y) для большого массива, разбитого на CHUNKS.
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

constexpr int CHUNKS  = 4;      // количество чанков/очередей
constexpr int THREADS = 256;    // для конфигурации ядра (в OpenCL — глобальный/локальный размер)

const char* kernelSrc = R"CLC(
__kernel void saxpy(float a, __global const float* x, __global float* y) {
    int idx = get_global_id(0);
    y[idx] = a * x[idx] + y[idx];
}
)CLC";

static void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "%s: %d\n", msg, err);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    size_t totalMB = 64; // 64 MiB по умолчанию
    if (argc == 2) totalMB = std::atoi(argv[1]);
    size_t totalBytes = totalMB * (1 << 20);
    size_t totalElems = totalBytes / sizeof(float);

    size_t elemsPerChunk = (totalElems + CHUNKS - 1) / CHUNKS;
    size_t bytesPerChunk = elemsPerChunk * sizeof(float);

    std::vector<float> h_x(totalElems, 1.0f), h_y(totalElems, 2.0f);

    cl_int err;
    cl_platform_id platform; check(clGetPlatformIDs(1, &platform, nullptr), "platform");
    cl_device_id device; check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "device");
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); check(err, "context");

    // Создаём независимые очереди команд (in-order). Для максимального оверлапа можно CL_QUEUE_OUT_OF_ORDER, но не все девайсы поддерживают.
    std::vector<cl_command_queue> queues(CHUNKS);
    for (int i = 0; i < CHUNKS; ++i) {
        queues[i] = clCreateCommandQueue(ctx, device, 0, &err); check(err, "queue");
    }

    // Общие буферы на устройстве (по chunk-адресованию)
    cl_mem d_x = clCreateBuffer(ctx, CL_MEM_READ_WRITE, totalBytes, nullptr, &err); check(err, "buf x");
    cl_mem d_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE, totalBytes, nullptr, &err); check(err, "buf y");

    // Программа + kernel
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernelSrc, nullptr, &err); check(err, "program");
    check(clBuildProgram(prog, 1, &device, "", nullptr, nullptr), "build");
    cl_kernel kernel = clCreateKernel(prog, "saxpy", &err); check(err, "kernel");

    float a = 2.0f;
    check(clSetKernelArg(kernel, 0, sizeof(float), &a), "arg0");

    auto t0 = std::chrono::high_resolution_clock::now();

    // Обрабатываем чанки
    for (int c = 0; c < CHUNKS; ++c) {
        size_t offsetElem = c * elemsPerChunk;
        size_t offsetByte = offsetElem * sizeof(float);
        size_t thisElems  = (c == CHUNKS - 1) ? (totalElems - offsetElem) : elemsPerChunk;
        size_t thisBytes  = thisElems * sizeof(float);
        cl_command_queue q = queues[c];

        // H2D копии
        check(clEnqueueWriteBuffer(q, d_x, CL_FALSE, offsetByte, thisBytes, h_x.data() + offsetElem, 0, nullptr, nullptr), "write x");
        check(clEnqueueWriteBuffer(q, d_y, CL_FALSE, offsetByte, thisBytes, h_y.data() + offsetElem, 0, nullptr, nullptr), "write y");

        // kernel
        check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_x), "arg1");
        check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_y), "arg2");

        size_t global = thisElems;
        check(clEnqueueNDRangeKernel(q, kernel, 1, &offsetElem, &global, nullptr, 0, nullptr, nullptr), "kernel");

        // D2H копия
        check(clEnqueueReadBuffer(q, d_y, CL_FALSE, offsetByte, thisBytes, h_y.data() + offsetElem, 0, nullptr, nullptr), "read y");
    }

    // Синхронизация всех очередей
    for (auto q : queues) clFinish(q);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Валидация нескольких элементов
    bool ok = true;
    for (int i = 0; i < 5; ++i) {
        size_t idx = (i * 997) % totalElems;
        if (std::fabs(h_y[idx] - (a * 1.0f + 2.0f)) > 1e-4f) { ok = false; break; }
    }

    std::printf("OpenCL overlap | Data %.2f MB, chunks %d, time %.2f ms, %s\n",
                totalBytes / 1e6, CHUNKS, ms_total, ok ? "PASSED" : "FAILED");

    clReleaseMemObject(d_x); clReleaseMemObject(d_y);
    clReleaseKernel(kernel); clReleaseProgram(prog);
    for (auto q : queues) clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return ok ? 0 : 1;
} 