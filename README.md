#CUDA Kernel Framework

Custom CUDA Frameowrk/Kernel, applied in deep belief networks. 


Mainly operates on CUDA Sorting Algorithms. CDN incorporated into code. PYTORCH, Tensor is used. 

CDN Example:
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


(ignore index.js, that was a test.)
