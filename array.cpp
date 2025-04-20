#include <CUDA-runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

template <typename CUDAType, typename HostType>
class CUDATraits {
public:
    using HostType = HostType;
    using DeviceType = CUDAType;
};
    using DevicePtrType = CUDAType*;
    using HostPtrType = HostType*;
    using SizeType = size_t;
    using ValueType = HostType;
{
    private:
        void setCudaArray(HostType* data, size_t size) {
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
        }
        DevicePtrType device_ptr_;
        HostPtrType host_ptr_;
        SizeType size_;

    public:
        void set(HostType* data, size_t size) {
            setCudaArray(data, size);
        }
        void get(HostType* data, size_t size) {
            getCudaArray(data, size);
        }

        DevicePtrType devicePtr() { return device_ptr_; }
        HostPtrType hostPtr() { return host_ptr_; }

        return 0;





}