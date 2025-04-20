#include <stdio.h>

// Simple CUDA kernel to add two vectors
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Array size (1M elements)
    int n = 1 << 20;
    size_t bytes = n * sizeof(int);
    
    // Host pointers
    int *h_a, *h_b, *h_c;
    
    // Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
    
    // Initialize host arrays
    for(int i = 0; i < n; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }
    
    // Device pointers
    int *d_a, *d_b, *d_c;
    
    // Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Threads per block
    int blockSize = 256;
    
    // Blocks per grid
    int gridSize = (n + blockSize - 1) / blockSize;
    
    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    for(int i = 0; i < n; i++) {
        if(h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d!\n", i);
            exit(1);
        }
    }
    printf("Vector addition successful!\n");
    
    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}