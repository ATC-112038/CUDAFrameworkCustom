#pragma once
#include <cstddef>

// Forward declaration of CNN operations
void cuda_conv2d_forward(
    const float* input, const float* filter, float* output,
    size_t batch_size, size_t in_channels, size_t out_channels,
    size_t input_height, size_t input_width,
    size_t kernel_size, size_t stride, size_t padding);

void cuda_conv2d_backward(
    const float* input, const float* filter, const float* grad_output,
    float* grad_input, float* grad_filter,
    size_t batch_size, size_t in_channels, size_t out_channels,
    size_t input_height, size_t input_width,
    size_t kernel_size, size_t stride, size_t padding);