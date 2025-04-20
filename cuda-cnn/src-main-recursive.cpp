#include "vector_ops.h"
#include <iostream>

int main() {
    const size_t N = 1 << 24; // 16M elements
    std::cout << "Vector Operations Benchmark (N = " << N << ")\n";
    
    // Allocate unified memory
    float* a = VectorOperations::allocateUnifiedMemory(N);
    float* b = VectorOperations::allocateUnifiedMemory(N);
    float* c = VectorOperations::allocateUnifiedMemory(N);
    
    // Initialize vectors
    VectorOperations::fillRandom(a, N);
    VectorOperations::fillRandom(b, N);
    
    // Perform operations
    {
        Timer timer;
        VectorOperations::unifiedVectorAdd(a, b, c, N);
        std::cout << "Vector addition: " << timer.elapsed() * 1000 << " ms\n";
    }
    
    // Verify results
    if (!VectorOperations::verifyResults(a, b, c, N)) {
        std::cerr << "Verification failed!\n";
        return 1;
    }
    
    // Clean up
    VectorOperations::freeUnifiedMemory(a);
    VectorOperations::freeUnifiedMemory(b);
    VectorOperations::freeUnifiedMemory(c);
    
    return 0;
}