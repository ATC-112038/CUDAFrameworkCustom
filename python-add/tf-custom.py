import tensorflow as tf
from tensorflow.python.framework import ops
from .core import vector_add, dot_product

def _tf_vector_add(a, b):
    a_np = a.numpy()
    b_np = b.numpy()
    return vector_add(a_np, b_np)

def tf_vector_add(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """TensorFlow CUDA vector addition"""
    return tf.py_function(_tf_vector_add, [a, b], tf.float32)

def _tf_dot_product(a, b):
    a_np = a.numpy()
    b_np = b.numpy()
    return dot_product(a_np, b_np)

def tf_dot_product(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """TensorFlow CUDA dot product"""
    return tf.py_function(_tf_dot_product, [a, b], tf.float32)

# Register gradients
@ops.RegisterGradient("TfVectorAdd")
def _vector_add_grad(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]
    return grad, grad

@ops.RegisterGradient("TfDotProduct")
def _dot_product_grad(op, grad):
    a = op.inputs[0]
    b = op.inputs[1]
    return grad * b, grad * a