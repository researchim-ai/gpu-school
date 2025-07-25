/*
 * hello_opencl.cpp — минимальный «Hello GPU» на OpenCL.
 * Цель примера — показать полный путь от обнаружения платформы до выполнения
 * самого простого ядра. Комментарии ⚑ отмечают ключевые этапы.
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

//⚑ 1. Исходный код устройства (OpenCL C)
//  Один kernel `set42`, который просто записывает число 42 в нулевой элемент
//  переданного буфера. R"CLC( ... )CLC" — сырой литерал-строка, чтобы избежать
//  экранирования символов.
const char* kernelSrc = R"CLC(
__kernel void set42(__global float* data) {
    data[0] = 42.0f; // операция на устройстве (GPU)
}
)CLC";

//⚑ 2. check() — маленький helper: обрываем программу, если любой вызов OpenCL
//  вернул не-нулевой код. Это избавляет от повторяющегося if(err!=CL_SUCCESS).
static void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "%s: %d\n", msg, err);
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    cl_int err;

    //⚑ 3. Получаем список платформ (драйверов) -----------------------------
    cl_uint numPlatforms = 0;
    check(clGetPlatformIDs(0, nullptr, &numPlatforms), "platform count");
    if (numPlatforms == 0) {
        std::fprintf(stderr, "No OpenCL platforms found\n");
        return EXIT_FAILURE;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    check(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr), "platform list");

    std::printf("Found %u OpenCL platform(s):\n", numPlatforms);
    for (cl_uint i = 0; i < numPlatforms; ++i) {
        char name[128] = {0};
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        std::printf("  [%u] %s\n", i, name);
    }

    //⚑ 4. Выбираем устройство (GPU) --------------------------------------
    cl_device_id device = nullptr;
    check(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "device");

    char devName[128] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(devName), devName, nullptr);
    std::printf("Using device: %s\n", devName);

    //⚑ 5. Создаём контекст и очередь команд ------------------------------
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); check(err, "context");
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err); check(err, "queue");

    //⚑ 6. Создаём и строим программу из исходника ------------------------
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernelSrc, nullptr, &err); check(err, "program");
    check(clBuildProgram(prog, 1, &device, "", nullptr, nullptr), "build");

    cl_kernel kernel = clCreateKernel(prog, "set42", &err); check(err, "kernel");

    //⚑ 7. Выделяем буфер и задаём аргументы ядра -------------------------
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float), nullptr, &err); check(err, "buffer");
    check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf), "arg0");

    //⚑ 8. Запускаем одну work-item --------------------------------------
    size_t global = 1; // глобальный размер NDRange
    check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "enqueue");
    check(clFinish(queue), "finish"); // ждём выполнения

    //⚑ 9. Читаем результат ----------------------------------------------
    float result = 0.0f;
    check(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(float), &result, 0, nullptr, nullptr), "read");
    std::printf("Result from GPU: %.1f (expected 42.0)\n", result);

    //⚑ 10. Очистка ресурсов ---------------------------------------------
    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return (result == 42.0f) ? 0 : 1;
} 