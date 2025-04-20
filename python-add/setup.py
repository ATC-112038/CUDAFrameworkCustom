from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Path to CUDA kernels
cuda_kernels = os.path.join('..', 'cuda_kernels')

extensions = [
    Extension(
        "cuda_ml.core",
        sources=["cuda_ml/core.pyx", 
                os.path.join(cuda_kernels, 'src', 'vector_ops.cu')],
        include_dirs=[os.path.join(cuda_kernels, 'include')],
        library_dirs=['/usr/local/cuda/lib64'],
        libraries=['cudart'],
        extra_compile_args=['-std=c++14', '-O3'],
        language='c++',
    )
]

setup(
    name="cuda_ml",
    version="0.1",
    packages=["cuda_ml"],
    ext_modules=cythonize(extensions),
    install_requires=[
        'numpy>=1.19',
        'torch>=1.8',
        'tensorflow>=2.4',
        'cython>=0.29'
    ],
)