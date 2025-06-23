/*
 * matmul_naive_cl.cpp — наивное матричное умножение C=A*B размером N×N на OpenCL.
 * Показывает работу с 2-D NDRange и проверку корректности.
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

const char* kernelSrc = R"CLC(
__kernel void matmul(const int N,
                     __global const float* A,
                     __global const float* B,
                     __global float* C) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        sum += A[row * N + k] * B[k * N + col];

    C[row * N + col] = sum;
}
)CLC";

static void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "%s: %d\n", msg, err);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int N = 256; // Default size
    if (argc == 2) N = std::atoi(argv[1]);
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, 0.0f);
    // Init A as identity, B with incremental numbers => C should equal B
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = (i == j) ? 1.0f : 0.0f;
            h_B[i * N + j] = static_cast<float>(i * N + j);
        }
    }

    cl_int err;
    cl_platform_id platform;
    check(clGetPlatformIDs(1, &platform, nullptr), "platform");
    cl_device_id device;
    check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "device");

    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); check(err, "context");
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err); check(err, "queue");

    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernelSrc, nullptr, &err); check(err, "program");
    check(clBuildProgram(prog, 1, &device, "", nullptr, nullptr), "build");

    cl_kernel kernel = clCreateKernel(prog, "matmul", &err); check(err, "kernel");

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A.data(), &err); check(err, "buf A");
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B.data(), &err); check(err, "buf B");
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err); check(err, "buf C");

    check(clSetKernelArg(kernel, 0, sizeof(int), &N), "arg0");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_A), "arg1");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_B), "arg2");
    check(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_C), "arg3");

    const size_t global[2] = { static_cast<size_t>(N), static_cast<size_t>(N) };

    const auto t0 = std::chrono::high_resolution_clock::now();
    check(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr), "enqueue");
    check(clFinish(queue), "finish");
    const auto t1 = std::chrono::high_resolution_clock::now();

    check(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, h_C.data(), 0, nullptr, nullptr), "read");

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Verify: since A = I, C should equal B
    bool ok = true;
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(h_C[i] - h_B[i]) > 1e-4f) { ok = false; break; }
    }

    std::printf("OpenCL MatMul %s | N=%d, time=%.3f ms\n", ok ? "PASSED" : "FAILED", N, ms);

    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(prog);
    clReleaseCommandQueue(queue); clReleaseContext(ctx);
    return ok ? 0 : 1;
} 