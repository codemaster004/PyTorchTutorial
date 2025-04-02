import torch
import torch.nn as nn


class LinearNN(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dim):
		super().__init__()
		
		# nn.Linear: y = xA^T + b
		self.layers = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, output_dim),
			nn.Tanh()
		)
	
	def forward(self, x):
		x = self.layers(x)
		
		return x
