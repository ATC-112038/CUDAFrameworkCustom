import tensorflow as tf
from tensorflow.python.framework import ops
from .core import _load_cuda_kernels

def tf_conv2d(input, filter, strides=1, padding='SAME'):
    """TensorFlow CUDA Conv2D operation"""
    if padding == 'SAME':
        padding_size = filter.shape[0] // 2
    else:
        padding_size = 0
    
    @tf.custom_gradient
    def _conv2d_forward(input, filter):
        def grad_fn(grad_output):
            # Call CUDA backward pass
            grad_input, grad_filter = _load_cuda_kernels().tf_conv2d_backward(
                input.numpy(), filter.numpy(), grad_output.numpy(),
                strides, padding_size)
            return tf.convert_to_tensor(grad_input), tf.convert_to_tensor(grad_filter)
        
        # Call CUDA forward pass
        output = _load_cuda_kernels().tf_conv2d_forward(
            input.numpy(), filter.numpy(),
            strides, padding_size)
        
        return tf.convert_to_tensor(output), grad_fn
    
    return _conv2d_forward(input, filter)