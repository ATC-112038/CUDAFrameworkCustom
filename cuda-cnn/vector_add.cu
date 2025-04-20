#include "vector_add.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

__global__ void vectorAddKernel(const int *d_a, const int *d_b, int *d_c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        d_c[tid] = d_a[tid] + d_b[tid];
    }
}

void vectorAdd(const int *h_a, const int *h_b, int *h_c, int n) {
    int *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(int);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, bytes));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Calculate and print execution time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel execution time: " << elapsed.count() * 1000 << " ms\n";

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
}

void initVectors(int *h_a, int *h_b, int n) {
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }
}

void verifyResult(const int *h_a, const int *h_b, const int *h_c, int n) {
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(1);
        }
    }
    std::cout << "Result verification successful!\n";
}