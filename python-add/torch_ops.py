import torch
from torch.autograd import Function
from .core import vector_add, dot_product

class CudaVectorAdd(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        result = vector_add(a_np, b_np)
        return torch.from_numpy(result).to(a.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output, grad_output

def torch_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch CUDA vector addition"""
    return CudaVectorAdd.apply(a, b)

class CudaDotProduct(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        return torch.tensor(dot_product(a_np, b_np), device=a.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output * b
        grad_b = grad_output * a
        return grad_a, grad_b

def torch_dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch CUDA dot product"""
    return CudaDotProduct.apply(a, b)