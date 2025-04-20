import torch
from torch.autograd import Function
from .core import _load_cuda_kernels

class CudaLSTMCell(Function):
    @staticmethod
    def forward(ctx, input, hidden, weight_ih, weight_hh, bias):
        ctx.save_for_backward(input, hidden, weight_ih, weight_hh, bias)
        h_prev, c_prev = hidden
        
        batch_size = input.size(0)
        hidden_size = h_prev.size(1)
        
        # Allocate outputs
        h_next = torch.empty_like(h_prev)
        c_next = torch.empty_like(c_prev)
        
        # Call CUDA LSTM kernel
        _load_cuda_kernels().lstm_forward(
            input, h_prev, c_prev,
            torch.cat([weight_ih, weight_hh], dim=1),
            bias,
            h_next, c_next,
            batch_size, input.size(1), hidden_size)
            
        return h_next, c_next

    @staticmethod
    def backward(ctx, grad_h_next, grad_c_next):
        input, hidden, weight_ih, weight_hh, bias = ctx.saved_tensors
        h_prev, c_prev = hidden
        
        # Allocate gradients
        grad_input = torch.zeros_like(input)
        grad_h_prev = torch.zeros_like(h_prev)
        grad_c_prev = torch.zeros_like(c_prev)
        grad_weight = torch.zeros_like(torch.cat([weight_ih, weight_hh], dim=1))
        grad_bias = torch.zeros_like(bias)
        
        # Call CUDA backward kernel
        _load_cuda_kernels().lstm_backward(
            input, h_prev, c_prev,
            torch.cat([weight_ih, weight_hh], dim=1),
            grad_h_next, grad_c_next,
            grad_input, grad_h_prev, grad_c_prev,
            grad_weight, grad_bias,
            input.size(0), input.size(1), h_prev.size(1))
            
        # Split weight gradients
        grad_weight_ih = grad_weight[:, :input.size(1)]
        grad_weight_hh = grad_weight[:, input.size(1):]
        
        return grad_input, (grad_h_prev, grad_c_prev), grad_weight_ih, grad_weight_hh, grad_bias

def lstm_cell(input, hidden, weight_ih, weight_hh, bias):
    """PyTorch CUDA LSTM Cell"""
    return CudaLSTMCell.apply(input, hidden, weight_ih, weight_hh, bias)