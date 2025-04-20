#include "cnn_ops.h"
#include <cuda_runtime.h>

__global__ void conv2d_forward_kernel(
    const float* input, const float* filter, float* output,
    int batch_size, int in_channels, int out_channels,
    int input_h, int input_w,
    int kernel_size, int stride, int padding) {
    
    // Output dimensions
    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;
    
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < batch_size * out_channels * output_h * output_w;
         idx += blockDim.x * gridDim.x) {
        
        // Decompose index
        int b = idx / (out_channels * output_h * output_w);
        int oc = (idx / (output_h * output_w)) % out_channels;
        int oh = (idx / output_w) % output_h;
        int ow = idx % output_w;
        
        float sum = 0.0f;
        
        // Loop over input channels and kernel
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int ih = oh * stride - padding + kh;
                    int iw = ow * stride - padding + kw;
                    
                    if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                        int input_idx = ((b * in_channels + ic) * input_h + ih) * input_w + iw;
                        int filter_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                        sum += input[input_idx] * filter[filter_idx];
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

// Implementation of cuda_conv2d_forward would call the kernel
// Similar implementations for backward pass would follow