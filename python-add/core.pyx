# distutils: language = c++
# distutils: sources = ../../cuda_kernels/src/vector_ops.cu
# distutils: extra_compile_args = -std=c++14 -O3
# distutils: include_dirs = ../../cuda_kernels/include

from libc.stdint cimport size_t

cdef extern from "cuda_interface.h":
    void cuda_vector_add(const float* a, const float* b, float* c, size_t n)
    float cuda_dot_product(const float* a, const float* b, size_t n)

import numpy as np

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Python interface for CUDA vector addition"""
    cdef size_t n = a.shape[0]
    cdef float[:] a_view = a.astype(np.float32, copy=False)
    cdef float[:] b_view = b.astype(np.float32, copy=False)
    cdef float[:] result = np.empty(n, dtype=np.float32)
    
    cuda_vector_add(&a_view[0], &b_view[0], &result[0], n)
    return np.asarray(result)

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Python interface for CUDA dot product"""
    cdef size_t n = a.shape[0]
    cdef float[:] a_view = a.astype(np.float32, copy=False)
    cdef float[:] b_view = b.astype(np.float32, copy=False)
    
    return cuda_dot_product(&a_view[0], &b_view[0], n)