import torch
from cuda_ml.cnn_ops import conv2d

# Create random input and filters
input = torch.randn(32, 3, 128, 128, device='cuda')  # batch, channels, h, w
weight = torch.randn(64, 3, 3, 3, device='cuda')      # out_channels, in_channels, h, w

# Perform convolution
output = conv2d(input, weight, stride=1, padding=1)
print("Output shape:", output.shape)  # Should be [32, 64, 128, 128]