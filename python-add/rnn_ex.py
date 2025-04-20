import torch
from cuda_ml.rnn_ops import lstm_cell

# LSTM parameters
input_size = 512
hidden_size = 256
batch_size = 32

# Create tensors
input = torch.randn(batch_size, input_size, device='cuda')
h_prev = torch.zeros(batch_size, hidden_size, device='cuda')
c_prev = torch.zeros(batch_size, hidden_size, device='cuda')
weight_ih = torch.randn(4*hidden_size, input_size, device='cuda')
weight_hh = torch.randn(4*hidden_size, hidden_size, device='cuda')
bias = torch.zeros(4*hidden_size, device='cuda')

# Forward pass
h_next, c_next = lstm_cell(input, (h_prev, c_prev), weight_ih, weight_hh, bias)
print("h_next shape:", h_next.shape)  # Should be [32, 256]