#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

__global__ void sortKernel(int *d_array, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < n; i++) {
        for (int j = tid; j < n - 1; j += blockDim.x * gridDim.x) {
            if (d_array[j] > d_array[j + 1]) {
                int temp = d_array[j];
                d_array[j] = d_array[j + 1];
                d_array[j + 1] = temp;
            }
        }
        __syncthreads();
    }
}

void printArray(const int *array, int n) {
    for (int i = 0; i < n; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    const int n = 10;
    int h_array[n] = {9, 7, 5, 3, 1, 2, 4, 6, 8, 0};
    int *d_array;

    std::cout << "Original array: ";
    printArray(h_array, n);

    cudaMalloc(&d_array, n * sizeof(int));
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    sortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted array: ";
    printArray(h_array, n);

    cudaFree(d_array);
    return 0;
}