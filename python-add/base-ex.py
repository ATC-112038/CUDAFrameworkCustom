import torch
from cuda_ml.torch_ops import torch_vector_add, torch_dot_product

# Create tensors (works on CPU or CUDA)
a = torch.rand(1000000, device='cuda')
b = torch.rand(1000000, device='cuda')

# Use our custom CUDA ops
result_add = torch_vector_add(a, b)
result_dot = torch_dot_product(a, b)

print("Vector add result:", result_add[:10])
print("Dot product result:", result_dot.item())