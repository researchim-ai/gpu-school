/*
 * saxpy_cl.cpp — кроссплатформенный пример SAXPY (Y = a*X + Y) на OpenCL.
 * Позволяет сравнить API OpenCL с CUDA.
 */

#define CL_TARGET_OPENCL_VERSION 120 // хотим API 1.2
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

const char* kernelSrc = R"CLC(
__kernel void saxpy(float a, __global const float* x, __global float* y) {
    int idx = get_global_id(0); // глобальный индекс нити
    y[idx] = a * x[idx] + y[idx];
}
)CLC";

static void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: %d\n", msg, err);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    size_t N = 1 << 20;              // 1M элементов
    if (argc == 2) N = std::atoi(argv[1]);
    size_t bytes = N * sizeof(float); // объём буфера в байтах

    std::vector<float> h_x(N, 1.0f), h_y(N, 2.0f);

    cl_int err;
    cl_platform_id platform;          // платформа OpenCL
    check(clGetPlatformIDs(1, &platform, nullptr), "platform");

    cl_device_id device;              // GPU-девайс
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "device");

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); // создаём контекст
    check(err, "context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err); // очередь команд
    check(err, "queue");

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSrc, nullptr, &err); // создаём программу из исходника
    check(err, "program");

    check(clBuildProgram(program, 1, &device, "", nullptr, nullptr), "build");

    cl_kernel kernel = clCreateKernel(program, "saxpy", &err); // берём кернел
    check(err, "kernel");

    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_x.data(), &err); // буфер X на GPU
    check(err, "buffer x");
    cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes, h_y.data(), &err); // буфер Y
    check(err, "buffer y");

    float a = 2.0f;
    check(clSetKernelArg(kernel, 0, sizeof(float), &a), "arg0");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_x), "arg1");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_y), "arg2");

    size_t global = N;               // глобальное количество work-items
    auto t0 = std::chrono::high_resolution_clock::now();
    check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "enqueue"); // запуск ядра
    check(clFinish(queue), "finish"); // ждём завершения
    auto t1 = std::chrono::high_resolution_clock::now();

    check(clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, bytes, h_y.data(), 0, nullptr, nullptr), "read"); // читаем результат

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool ok = true; // флаг корректности
    for (size_t i = 0; i < N; ++i) if (fabs(h_y[i] - (a * 1.0f + 2.0f)) > 1e-5f){ ok=false; break; }

    printf("OpenCL SAXPY %s | N=%zu, time=%.3f ms\n", ok?"PASSED":"FAILED", N, ms);

    clReleaseMemObject(d_x); clReleaseMemObject(d_y); // освобождаем буферы
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    return ok?0:1;
} 