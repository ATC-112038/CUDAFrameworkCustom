#include "vector_ops.h"
#include <curand_kernel.h>
#include <cub/cub.cuh>

// Shared memory size for optimized kernels
constexpr int SHARED_MEM_SIZE = 256;

// Optimized vector addition kernel with loop unrolling
__global__ void vectorAddKernel(const float* __restrict__ a, 
                               const float* __restrict__ b, 
                               float* __restrict__ c, 
                               int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Loop unrolling for better performance
    #pragma unroll 4
    for (int i = tid; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Optimized vector multiplication kernel with shared memory
__global__ void vectorMultiplyKernel(const float* __restrict__ a, 
                                    const float* __restrict__ b, 
                                    float* __restrict__ c, 
                                    int n) {
    __shared__ float s_a[SHARED_MEM_SIZE];
    __shared__ float s_b[SHARED_MEM_SIZE];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (global_idx < n) {
        s_a[tid] = a[global_idx];
        s_b[tid] = b[global_idx];
        __syncthreads();
        
        c[global_idx] = s_a[tid] * s_b[tid];
    }
}

// Dot product kernel using parallel reduction
__global__ void dotProductKernel(const float* a, const float* b, float* temp, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? a[i] * b[i] : 0.0f;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        temp[blockIdx.x] = sdata[0];
    }
}

// SAXPY kernel (a*x + y)
__global__ void saxpyKernel(float a, const float* x, float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

// Implementations of VectorOperations methods
void VectorOperations::unifiedVectorAdd(float* a, float* b, float* c, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    vectorAddKernel<<<gridSize, blockSize>>>(a, b, c, n);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

// [Additional implementations would follow...]