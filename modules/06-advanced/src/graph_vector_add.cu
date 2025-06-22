/*****
 * graph_vector_add.cu — пример использования CUDA Graph API
 * для ускорения повторного запуска однотипных операций
 * (H2D копия → ядро VectorAdd → D2H копия).
 *
 * Идея: вместо того, чтобы каждый раз заново оформлять десятки вызовов
 * CUDA Runtime, мы один раз «записываем» граф, который сразу знает все
 * зависимости, и затем много раз исполняем его через `cudaGraphLaunch`.
 *
 * Граф особенно полезен в сценариях ML-инференса или медиапайплайна,
 * где последовательность операций фиксирована.
 *****/

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

constexpr int THREADS = 256;

/******************* Ядро сложения *******************/
__global__ void vecAddGraph(const float* a, const float* b, float* c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

static inline void check(cudaError_t e, const char* m) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int n = 1 << 20;                       // 1M элементов
    if (argc == 2) n = std::atoi(argv[1]);
    size_t bytes = n * sizeof(float);

    /***** 1. Host память *****/
    std::vector<float> h_a(n, 1.0f), h_b(n, 2.0f), h_c(n);

    /***** 2. Device память *****/
    float *d_a, *d_b, *d_c;
    check(cudaMalloc(&d_a, bytes), "malloc a");
    check(cudaMalloc(&d_b, bytes), "malloc b");
    check(cudaMalloc(&d_c, bytes), "malloc c");

    /***** 3. Создаём пустой граф *****/
    cudaStream_t captureStream;
    cudaStreamCreate(&captureStream);
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // --- Запись операций ---
    check(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal), "begin capture");

    // H2D копии
    cudaMemcpyAsync(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice, captureStream);
    cudaMemcpyAsync(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice, captureStream);

    // Запуск ядра
    int blocks = (n + THREADS - 1) / THREADS;
    vecAddGraph<<<blocks, THREADS, 0, captureStream>>>(d_a, d_b, d_c, n);

    // D2H копия
    cudaMemcpyAsync(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost, captureStream);

    check(cudaStreamEndCapture(captureStream, &graph), "end capture");

    // --- Инстанцируем граф ---
    check(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), "instantiate");

    /***** 4. Многократный запуск графа *****/
    const int iters = 100; // число повторов
    for (int i = 0; i < iters; ++i) {
        check(cudaGraphLaunch(graphExec, captureStream), "launch graph");
        check(cudaStreamSynchronize(captureStream), "sync");
    }

    /***** 5. Проверка корректности *****/
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(h_c[i] - 3.0f) > 1e-5f) { ok = false; break; }
    }
    printf("Graph VectorAdd %s | iters=%d, elements=%d\n", ok ? "PASSED" : "FAILED", iters, n);

    /***** 6. Очистка *****/
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(captureStream);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return ok ? 0 : 1;
} 