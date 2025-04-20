import tensorflow as tf
from cuda_ml.tf_ops import tf_vector_add, tf_dot_product

# Create tensors
a = tf.random.uniform([1000000])
b = tf.random.uniform([1000000])

# Use our custom CUDA ops
result_add = tf_vector_add(a, b)
result_dot = tf_dot_product(a, b)

print("Vector add result:", result_add[:10])
print("Dot product result:", result_dot.numpy())