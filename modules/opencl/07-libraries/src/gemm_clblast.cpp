/*
 * gemm_clblast.cpp — пример использования библиотеки CLBlast для умножения матриц
 * C = alpha·A×B + beta·C (SGEMM). Показывает, как вызывать высокоуровневые BLAS
 * функции поверх существующего OpenCL-контекста.
 */

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <clblast_c.h>   // C-интерфейс CLBlast
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>

static void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "%s: %d\n", msg, err);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int N = 512; // размер квадратных матриц
    if (argc == 2) N = std::atoi(argv[1]);
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    // --- 1. Хостовые данные
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, 0.0f), h_ref(N * N);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.f;
        h_B[i] = static_cast<float>((i * 7) % 100) / 100.f;
    }

    cl_int err;
    cl_platform_id platform; check(clGetPlatformIDs(1, &platform, nullptr), "platform");
    cl_device_id device; check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "device");
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); check(err, "ctx");
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err); check(err, "queue");

    // --- 2. Device buffers
    cl_mem d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_A.data(), &err); check(err, "A");
    cl_mem d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_B.data(), &err); check(err, "B");
    cl_mem d_C = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err); check(err, "C");

    // --- 3. Запуск SGEMM через CLBlast
    const float alpha = 1.0f, beta = 0.0f;
    const size_t ld = N;

    auto t0 = std::chrono::high_resolution_clock::now();
    CLBlastStatusCode status = CLBlastSgemm(
        CLBlastLayoutRowMajor,
        CLBlastTransposeNo, CLBlastTransposeNo,
        N, N, N,
        alpha,
        d_A, 0, ld,
        d_B, 0, ld,
        beta,
        d_C, 0, ld,
        &queue, nullptr);
    if (status != CLBlastSuccess) {
        std::fprintf(stderr, "CLBlastSgemm error: %d\n", status);
        return 1;
    }
    clFinish(queue);
    auto t1 = std::chrono::high_resolution_clock::now();

    // --- 4. Копируем результат и валидируем (CPU для небольших N)
    check(clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, bytes, h_C.data(), 0, nullptr, nullptr), "read C");

    bool ok = true;
    if (N <= 512) {
        // CPU reference
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                float sum = 0.f;
                for (int k = 0; k < N; ++k) sum += h_A[r * N + k] * h_B[k * N + c];
                h_ref[r * N + c] = sum;
            }
        for (int i = 0; i < N * N; ++i) if (std::fabs(h_C[i] - h_ref[i]) > 1e-3f) { ok = false; break; }
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("CLBlast SGEMM %s | N=%d, time=%.2f ms\n", ok ? "PASSED" : "FAILED", N, ms);

    clReleaseMemObject(d_A); clReleaseMemObject(d_B); clReleaseMemObject(d_C);
    clReleaseCommandQueue(queue); clReleaseContext(ctx);
    return ok ? 0 : 1;
} 