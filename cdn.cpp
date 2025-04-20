#include <iostream>
#include <cuda_runtime.h>

// Kernel function to demonstrate low-level CUDA interaction
__global__ void lowLevelKernel(int *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] *= 2; // Example operation: double the value
    }
}

// High-level function to manage CUDA operations
void highLevelInteraction(int *hostData, int size) {
    int *deviceData;
    size_t dataSize = size * sizeof(int);

    // Allocate memory on the device
    cudaMalloc((void **)&deviceData, dataSize);

    // Copy data from host to device
    cudaMemcpy(deviceData, hostData, dataSize, cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    lowLevelKernel<<<blocksPerGrid, threadsPerBlock>>>(deviceData, size);

    // Copy results back to host
    cudaMemcpy(hostData, deviceData, dataSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceData);
}

int main() {
    const int size = 10;
    int hostData[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::cout << "Original data: ";
    for (int i = 0; i < size; ++i) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;

    // High-level interaction with CUDA
    highLevelInteraction(hostData, size);

    std::cout << "Processed data: ";
    for (int i = 0; i < size; ++i) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}