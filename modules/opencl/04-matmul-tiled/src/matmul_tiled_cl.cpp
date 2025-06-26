/*
 * matmul_tiled_cl.cpp — матричное умножение C=A*B с тайлингом 16×16 и использованием локальной памяти.
 * Демонстрирует оптимизацию пропускной способности памяти на OpenCL.
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

constexpr int TILE = 16;

const char* kernelSrc = R"CLC(
#define TILE 16
__kernel void matmul_tiled(const int N,
                           __global const float* A,
                           __global const float* B,
                           __global float* C) {
    const int row = get_global_id(1); // Y
    const int col = get_global_id(0); // X

    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];

    float sum = 0.0f;
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int tiledRow = row;
        int tiledCol = t * TILE + get_local_id(0);
        As[get_local_id(1)][get_local_id(0)] = (tiledRow < N && tiledCol < N) ?
            A[tiledRow * N + tiledCol] : 0.0f;

        tiledRow = t * TILE + get_local_id(1);
        tiledCol = col;
        Bs[get_local_id(1)][get_local_id(0)] = (tiledRow < N && tiledCol < N) ?
            B[tiledRow * N + tiledCol] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; ++k)
            sum += As[get_local_id(1)][k] * Bs[k][get_local_id(0)];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}
)CLC";

static void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "%s: %d\n", msg, err);
        std::exit(EXIT_FAILURE);
    }
}

// CPU reference for validation (optional small N)
static void matmulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            float s = 0.f;
            for (int k = 0; k < N; ++k)
                s += A[r * N + k] * B[k * N + c];
            C[r * N + c] = s;
        }
}

int main(int argc, char** argv) {
    int N = 512;
    if (argc == 2) N = std::atoi(argv[1]);
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N), h_ref(N * N);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.f;
        h_B[i] = static_cast<float>((i * 7) % 100) / 100.f;
    }

    cl_int err;
    cl_platform_id platform; check(clGetPlatformIDs(1, &platform, nullptr), "platform");
    cl_device_id device; check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "device");
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); check(err, "ctx");
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err); check(err, "queue");

    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernelSrc, nullptr, &err); check(err, "program");
    check(clBuildProgram(prog, 1, &device, "", nullptr, nullptr), "build");
    cl_kernel kernel = clCreateKernel(prog, "matmul_tiled", &err); check(err, "kernel");

    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A.data(), &err); check(err, "A");
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B.data(), &err); check(err, "B");
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err); check(err, "C");

    check(clSetKernelArg(kernel, 0, sizeof(int), &N), "arg0");
    check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_A), "arg1");
    check(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_B), "arg2");
    check(clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_C), "arg3");

    size_t local[2]  = { TILE, TILE };
    size_t global[2] = { static_cast<size_t>((N + TILE - 1) / TILE * TILE),
                         static_cast<size_t>((N + TILE - 1) / TILE * TILE) };

    auto t0 = std::chrono::high_resolution_clock::now();
    check(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 0, nullptr, nullptr), "enqueue");
    check(clFinish(queue), "finish");
    auto t1 = std::chrono::high_resolution_clock::now();

    check(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, h_C.data(), 0, nullptr, nullptr), "read");

    double ms_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count();

    bool validate = (N <= 512);
    bool ok = true;
    if (validate) {
        auto s0 = std::chrono::high_resolution_clock::now();
        matmulCPU(h_A, h_B, h_ref, N);
        auto s1 = std::chrono::high_resolution_clock::now();
        double ms_cpu = std::chrono::duration<double, std::milli>(s1 - s0).count();
        for (int i = 0; i < N * N; ++i) {
            if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) { ok = false; break; }
        }
        std::printf("Validation %s | CPU %.2f ms\n", ok ? "PASSED" : "FAILED", ms_cpu);
        if (ok) std::printf("Speedup: %.2fx\n", ms_cpu / ms_gpu);
    }

    std::printf("OpenCL tiled matmul %s | N=%d, GPU time=%.2f ms\n", ok ? "SUCCESS" : "ERROR", N, ms_gpu);

    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseKernel(kernel); clReleaseProgram(prog);
    clReleaseCommandQueue(queue); clReleaseContext(ctx);
    return ok ? 0 : 1;
} 