#include "vector_add.h"
#include <iostream>
#include <cstdlib>

int main() {
    // Array size (1M elements)
    const int n = 1 << 20;
    std::cout << "Vector size: " << n << " elements\n";

    // Allocate host memory
    int *h_a = new int[n];
    int *h_b = new int[n];
    int *h_c = new int[n];

    // Initialize vectors
    initVectors(h_a, h_b, n);

    // Run vector addition
    vectorAdd(h_a, h_b, h_c, n);

    // Verify results
    verifyResult(h_a, h_b, h_c, n);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}