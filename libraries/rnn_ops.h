#pragma once
#include <cstddef>

void cuda_lstm_forward(
    const float* input, const float* h_prev, const float* c_prev,
    const float* weights, const float* biases,
    float* h_next, float* c_next,
    size_t batch_size, size_t input_size, size_t hidden_size);

void cuda_lstm_backward(
    const float* input, const float* h_prev, const float* c_prev,
    const float* weights, const float* h_next, const float* c_next,
    const float* grad_h_next, const float* grad_c_next,
    float* grad_input, float* grad_h_prev, float* grad_c_prev,
    float* grad_weights, float* grad_biases,
    size_t batch_size, size_t input_size, size_t hidden_size);