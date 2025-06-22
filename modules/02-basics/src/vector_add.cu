#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

constexpr int THREADS = 256;

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int n = 1 << 20; // 1M elements
    if (argc == 2) n = std::atoi(argv[1]);

    size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n);

    float *d_a, *d_b, *d_c;
    checkCuda(cudaMalloc(&d_a, bytes), "malloc a");
    checkCuda(cudaMalloc(&d_b, bytes), "malloc b");
    checkCuda(cudaMalloc(&d_c, bytes), "malloc c");

    checkCuda(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice), "copy a");
    checkCuda(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice), "copy b");

    int blocks = (n + THREADS - 1) / THREADS;
    vecAdd<<<blocks, THREADS>>>(d_a, d_b, d_c, n);
    checkCuda(cudaGetLastError(), "kernel");

    checkCuda(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "copy c back");

    // простая проверка
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(h_c[i] - 3.0f) > 1e-5f) { ok = false; break; }
    }
    printf("VectorAdd %s for %d elements\n", ok ? "PASSED" : "FAILED", n);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return ok ? 0 : 1;
} 