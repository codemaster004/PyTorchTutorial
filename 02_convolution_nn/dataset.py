import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class CNNDataset(Dataset):
	def __init__(self, dataset_path, transform=None):
		# Load the dataset, assumed that it wa created by download.py
		data = pd.read_csv(dataset_path)
		self.transform = transform  # transformation for data optional
		
		self.paths = data['path']  # Copy raw paths
		# Labels need to be integers, so we first need to specify way to connect label with number
		self.labels_lookup = sorted(data['label'].unique().tolist())  # index in table of alphabetical order should work.
		self.labels = data['label'].apply(lambda x: self.labels_lookup.index(x))  # now just replace raw label with correct index
	
	def __len__(self):
		return self.paths.shape[0]  # Length of entire dataset is length of path column
	
	def __getitem__(self, idx):
		image = Image.open(self.paths[idx]).convert('RGB')  # read the image in RGB
		# If there is a transformation specify, transform the image
		if self.transform:
			image = self.transform(image)
		# convert the image from PIL format into a numpy array
		image = np.array(image).astype(np.uint8)  # Output: H x W x C = 3
		# Neural networks like input that are close to zero, but from numpy we get matrix of types int [0, 255]
		image = (image / 127.5 - 1.0).astype(np.float32)  # Converts the matrix to type float [-1, 1]
		image = torch.from_numpy(image).permute(2, 0, 1)  # PyTorch require input matrix for ConvLayers to be C x H x W
		
		return image, self.labels[idx]  # return (torch ready image, correct label)
