import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	def __init__(self, num_classes):
		super(CNN, self).__init__()
		
		self.feature_extract = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			
			nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			
			nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(1),
		)
		
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=128, out_features=num_classes),
		)
	
	def forward(self, x):
		x = self.feature_extract(x)
		x = self.classifier(x)
		return x
