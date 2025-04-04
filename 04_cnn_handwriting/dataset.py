import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


def transformation(image, mean_w, mean_h):
	aspect_ratio = image.width / image.height
	
	# Resize all images into a target height
	new_width = int(mean_h * aspect_ratio)
	image = image.resize((new_width, mean_h), Image.LANCZOS)
	
	background = Image.new("L", (mean_w, mean_h), (255,))
	if new_width < mean_w:
		left_padding = (mean_w - new_width) // 2
		background.paste(image, (left_padding, 0))
		image = background
	elif new_width > mean_w:
		image = image.resize((mean_w, mean_h))
	image.show()
	
	return image


class PLWordsDataset(Dataset):
	def __init__(self, dataset_path, transform=None):
		data = pd.read_csv(dataset_path)  # Load csv dataset info
		self.transform = transform  # transformation on images, optional
		self.paths = data['path']  # Copy raw paths
		self.positions = data[['x1', 'y1', 'x2', 'y2']]
		self.mean_width = int((self.positions['x2'] - self.positions['x1']).mean())
		self.mean_height = int((self.positions['y2'] - self.positions['y1']).mean())
		# Change labels to indexes, logits
		data['label'] = data['label'].astype(str)
		self.labels_lookup = sorted(
			data['label'].unique().tolist())  # index in table of alphabetical order
		self.labels = data['label'].apply(
			lambda x: self.labels_lookup.index(x))  # replace raw label with indexes
	
	def __len__(self):
		return self.paths.shape[0]  # Length of entire dataset is length of path column
	
	def __getitem__(self, idx):
		image = Image.open(self.paths[idx]).convert('L')  # read the image in Grayscale
		# Cropping to a single word
		to_crop = (self.positions['x1'][idx],
		           self.positions['y1'][idx],
		           self.positions['x2'][idx],
		           self.positions['y2'][idx])
		image = image.crop(to_crop)
		# If there is a transformation specify, transform the image
		if self.transform:
			image = self.transform(image, mean_w=self.mean_width, mean_h=self.mean_height)
		# convert the image from PIL format into a numpy array
		image = np.array(image).astype(np.uint8)  # Output: H x W x C = 3
		# Neural networks like input that are close to zero, but from numpy we get matrix of types int [0, 255]
		image = (image / 127.5 - 1.0).astype(np.float32)  # Converts the matrix to type float [-1, 1]
		image = torch.from_numpy(image)  # PyTorch require input matrix for ConvLayers to be C x H x W
		
		return image, self.labels[idx]  # return (torch ready image, correct label)


if __name__ == '__main__':
	dataset = PLWordsDataset("./data/dataset.csv", transform=transformation)
	# tensor = dataset[0]
	# tensor = dataset[36]
