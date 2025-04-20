#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cuda-runtime.h> 

template <typename CUDAType, typename HostType>
class CudaArray {
public:
    CudaArray(size_t size) : size_(size) {
        cudaMalloc((void**)&device_ptr_, size_ * sizeof(CUDAType));
        host_ptr_ = (HostType*)malloc(size_ * sizeof(HostType));
    }

    ~CudaArray() {
        cudaFree(device_ptr_);
        free(host_ptr_);
    }

    void copyToDevice() {
        cudaMemcpy(device_ptr_, host_ptr_, size_ * sizeof(CUDAType), cudaMemcpyHostToDevice);
    }

    void copyToHost() {
        cudaMemcpy(host_ptr_, device_ptr_, size_ * sizeof(CUDAType), cudaMemcpyDeviceToHost);
    }

    CUDAType* devicePtr() { return device_ptr_; }
    HostType* hostPtr() { return host_ptr_; }
    size_t size() { return size_; }

    public:
       void set CudaArray(HostType* data, size_t size) {
           if (size > size_) {
               free(host_ptr_);
               cudaFree(device_ptr_);
               size_ = size;
               cudaMalloc((void**)&device_ptr_, size_ * sizeof(CUDAType));
               host_ptr_ = (HostType*)malloc(size_ * sizeof(HostType));
           }
           memcpy(host_ptr_, data, size_ * sizeof(HostType));
       }

    void getCudaArray(HostType* data, size_t size) {
        if (size > size_) {
            free(host_ptr_);
            cudaFree(device_ptr_);
            size_ = size;
            cudaMalloc((void**)&device_ptr_, size_ * sizeof(CUDAType));
            host_ptr_ = (HostType*)malloc(size_ * sizeof(HostType));
        }
        memcpy(data, host_ptr_, size_ * sizeof(HostType));

        return 0;

    }
        