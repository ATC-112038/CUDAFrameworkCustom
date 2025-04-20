import torch
from torch.autograd import Function
from .core import _load_cuda_kernels

class CudaConv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, stride=1, padding=0):
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels = weight.shape[0]
        kernel_size = weight.shape[2]
        
        # Calculate output dimensions
        out_h = (in_h + 2 * padding - kernel_size) // stride + 1
        out_w = (in_w + 2 * padding - kernel_size) // stride + 1
        
        output = torch.empty(batch_size, out_channels, out_h, out_w,
                            device=input.device)
        
        # Call CUDA kernel
        _load_cuda_kernels().conv2d_forward(
            input, weight, output,
            batch_size, in_channels, out_channels,
            in_h, in_w, kernel_size, stride, padding)
            
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        
        # Allocate gradients
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        
        # Call CUDA backward kernel
        _load_cuda_kernels().conv2d_backward(
            input, weight, grad_output,
            grad_input, grad_weight,
            input.shape[0], input.shape[1], weight.shape[0],
            input.shape[2], input.shape[3],
            weight.shape[2], stride, padding)
            
        return grad_input, grad_weight, None, None

def conv2d(input, weight, stride=1, padding=0):
    """PyTorch CUDA Conv2d operation"""
    return CudaConv2d.apply(input, weight, stride, padding)