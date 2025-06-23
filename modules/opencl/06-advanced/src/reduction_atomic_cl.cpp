/*
 * reduction_atomic_cl.cpp — вычисление суммы (dot product) массива с использованием атомарной
 * операции `atomic_add` в глобальной памяти. Показывает применение более «продвинутых» примитивов
 * синхронизации OpenCL 2.0.
 */

#define CL_TARGET_OPENCL_VERSION 200 // нужна atomic_add(float *, float)
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <string>

constexpr int THREADS = 256;

// Ядро, использующее атомарные float (требует OpenCL 2.0 + расширение)
const char* atomicKernelSrc = R"CLC(
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

__kernel void reduce_atomic(__global const float* data, __global float* result, const int N) {
    int idx = get_global_id(0);
    float val = 0.0f;
    if (idx < N) val = data[idx];
    // atomic add в глобальной памяти (OpenCL 2.0+ для float)
    atomic_fetch_add_explicit((atomic_float*)result, val, memory_order_relaxed, memory_scope_device);
}
)CLC";

// Fallback ядро: редукция внутри work-group, запись частичных сумм в выходной массив
const char* wgKernelSrc = R"CLC(
__kernel void reduce_wg(__global const float* data,
                        __global float* partial,
                        const int N) {
    __local float sdata[256];
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group = get_group_id(0);

    float v = (gid < N) ? data[gid] : 0.0f;
    sdata[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (lid < stride) sdata[lid] += sdata[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[group] = sdata[0];
}
)CLC";

static void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "%s: %d\n", msg, err);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int N = 1 << 24; // ~16M элементов
    if (argc == 2) N = std::atoi(argv[1]);
    size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f; // ожидаем сумму = N

    cl_int err;
    cl_platform_id platform; check(clGetPlatformIDs(1, &platform, nullptr), "platform");
    cl_device_id device; check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "device");

    // Проверяем поддержку OpenCL 2.0
    char versionBuf[64];
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(versionBuf), versionBuf, nullptr);
    int major = 1, minor = 1;
    sscanf(versionBuf, "OpenCL %d.%d", &major, &minor);
    if (major < 2) {
        std::fprintf(stderr, "Device does not support OpenCL 2.0 atomics (reported %s)\n", versionBuf);
        return 0; // graceful exit
    }

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); check(err, "ctx");
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err); check(err, "queue");

    // --- Пытаемся скомпилировать атомарное ядро ---
    cl_int buildErr;
    const char* opts = "-cl-std=CL2.0";
    cl_program progAtomic = clCreateProgramWithSource(ctx, 1, &atomicKernelSrc, nullptr, &err); check(err, "progAtomic");
    buildErr = clBuildProgram(progAtomic, 1, &device, opts, nullptr, nullptr);

    bool atomicAvailable = (buildErr == CL_SUCCESS);

    if (!atomicAvailable) {
        // выводим лог для справки
        size_t logSize = 0; clGetProgramBuildInfo(progAtomic, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        if (logSize) {
            std::string log(logSize, '\0');
            clGetProgramBuildInfo(progAtomic, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::fprintf(stderr, "Atomic kernel build failed, falling back. Log:\n%s\n", log.c_str());
        }
        clReleaseProgram(progAtomic);
    }

    cl_mem d_data = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_data.data(), &err); check(err, "data");

    float gpuSum = 0.0f;
    double ms = 0.0;

    if (atomicAvailable) {
        // --- Однопроходовая атомарная редукция ---
        cl_mem d_sum  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float), nullptr, &err); check(err, "sum");
        float zero = 0.0f;
        check(clEnqueueWriteBuffer(queue, d_sum, CL_TRUE, 0, sizeof(float), &zero, 0, nullptr, nullptr), "init zero");

        cl_kernel kAtomic = clCreateKernel(progAtomic, "reduce_atomic", &err); check(err, "kAtomic");
        check(clSetKernelArg(kAtomic, 0, sizeof(cl_mem), &d_data), "arg0");
        check(clSetKernelArg(kAtomic, 1, sizeof(cl_mem), &d_sum), "arg1");
        check(clSetKernelArg(kAtomic, 2, sizeof(int), &N), "arg2");

        size_t global = ((N + THREADS - 1) / THREADS) * THREADS;
        size_t local  = THREADS;

        auto t0 = std::chrono::high_resolution_clock::now();
        check(clEnqueueNDRangeKernel(queue, kAtomic, 1, nullptr, &global, &local, 0, nullptr, nullptr), "enqueue atomic");
        check(clFinish(queue), "finish atomic");
        auto t1 = std::chrono::high_resolution_clock::now();

        check(clEnqueueReadBuffer(queue, d_sum, CL_TRUE, 0, sizeof(float), &gpuSum, 0, nullptr, nullptr), "read sum");
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        clReleaseMemObject(d_sum);
        clReleaseKernel(kAtomic);
        clReleaseProgram(progAtomic);
    } else {
        // --- Двухпроходовая редукция ---
        cl_program progWG = clCreateProgramWithSource(ctx, 1, &wgKernelSrc, nullptr, &err); check(err, "progWG");
        check(clBuildProgram(progWG, 1, &device, "", nullptr, nullptr), "build wg");
        cl_kernel kWG = clCreateKernel(progWG, "reduce_wg", &err); check(err, "kWG");

        size_t global = ((N + THREADS - 1) / THREADS) * THREADS;
        size_t local  = THREADS;
        size_t groups = global / local;

        cl_mem d_partial = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, groups * sizeof(float), nullptr, &err); check(err, "partial");

        check(clSetKernelArg(kWG, 0, sizeof(cl_mem), &d_data), "wg arg0");
        check(clSetKernelArg(kWG, 1, sizeof(cl_mem), &d_partial), "wg arg1");
        check(clSetKernelArg(kWG, 2, sizeof(int), &N), "wg arg2");

        auto t0 = std::chrono::high_resolution_clock::now();
        check(clEnqueueNDRangeKernel(queue, kWG, 1, nullptr, &global, &local, 0, nullptr, nullptr), "enqueue wg");
        check(clFinish(queue), "finish wg");
        auto t1 = std::chrono::high_resolution_clock::now();

        std::vector<float> partial(groups);
        check(clEnqueueReadBuffer(queue, d_partial, CL_TRUE, 0, groups * sizeof(float), partial.data(), 0, nullptr, nullptr), "read partial");

        gpuSum = 0.0f;
        for (float v : partial) gpuSum += v;
        ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        clReleaseMemObject(d_partial);
        clReleaseKernel(kWG);
        clReleaseProgram(progWG);
    }

    // CPU reference
    double cpuSum = static_cast<double>(N) * 1.0;
    bool ok = std::fabs(gpuSum - cpuSum) < 1e-3f;

    std::printf("OpenCL reduction %s | N=%d, sum=%.1f, time=%.2f ms (%s)\n",
                ok ? "PASSED" : "FAILED", N, gpuSum, ms,
                atomicAvailable ? "atomic" : "wg + host");

    clReleaseMemObject(d_data);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return ok ? 0 : 1;
} 