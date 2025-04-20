#pragma once
#include "cuda_utils.h"
#include <memory>

class VectorOperations {
public:
    // Unified memory vector operations
    static void unifiedVectorAdd(float* a, float* b, float* c, int n);
    static void unifiedVectorMultiply(float* a, float* b, float* c, int n);
    
    // Pinned memory operations with streams
    static void streamedVectorAdd(const float* h_a, const float* h_b, float* h_c, int n);
    static void streamedVectorMultiply(const float* h_a, const float* h_b, float* h_c, int n);
    
    // Advanced operations
    static void dotProduct(const float* a, const float* b, float* result, int n);
    static void saxpy(float a, const float* x, float* y, int n);
    
    // Memory management
    static float* allocateUnifiedMemory(size_t n);
    static float* allocatePinnedMemory(size_t n);
    static void freeUnifiedMemory(float* ptr);
    static void freePinnedMemory(float* ptr);
    
    // Utility functions
    static void fillRandom(float* data, int n, float min = 0.0f, float max = 1.0f);
    static bool verifyResults(const float* a, const float* b, const float* c, int n, float tolerance = 1e-6f);
};