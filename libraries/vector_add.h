#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <cuda_runtime.h>
#include <iostream>

// Function declarations
void vectorAdd(const int *h_a, const int *h_b, int *h_c, int n);
void initVectors(int *h_a, int *h_b, int n);
void verifyResult(const int *h_a, const int *h_b, const int *h_c, int n);

// CUDA error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#endif // VECTOR_ADD_H