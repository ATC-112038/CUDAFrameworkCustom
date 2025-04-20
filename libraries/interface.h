#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_vector_add(const float* a, const float* b, float* c, size_t n);
void cuda_vector_multiply(const float* a, const float* b, float* c, size_t n);
float cuda_dot_product(const float* a, const float* b, size_t n);
void cuda_saxpy(float alpha, const float* x, float* y, size_t n);

#ifdef __cplusplus
}
#endif