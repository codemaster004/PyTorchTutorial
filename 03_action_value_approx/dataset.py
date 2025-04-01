import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RLDataset(Dataset):
	def __init__(self, dataframe):
		self.states = dataframe[0]
		self.actions = dataframe[1]
		self.rewards = dataframe[2]
	
	def __len__(self):
		return self.actions.shape[0]
	
	def __getitem__(self, idx):
		state = self.states[idx]
		action = self.actions[idx]  # TODO: CONVERT THIS INTO A VECTOR
		
		x = np.array(np.append(state, action))
		x = torch.tensor(x, dtype=torch.float32)
		
		y = torch.tensor(np.array(self.rewards[idx]), dtype=torch.float32).flatten()
		
		return x, y
