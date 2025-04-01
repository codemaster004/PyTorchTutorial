import torch
import torch.nn as nn


class LinearNN(nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		
		# nn.Linear: y = xA^T + b
		self.linear = nn.Linear(input_dim, output_dim)
	
	def forward(self, x):
		x = torch.relu(self.linear(x))
		
		return x
