import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		# Define Max-Pooling layer, we define only one since it is not parametrized, it only always takes the max value.
		self.pool = nn.MaxPool2d(2, 2)  # Pooling layers are used to reduce the image size, and save on VRAM
		# Now we need to define a couple of Convolution layers, those are out model weights so probably we need more than one.
		self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		# Final layer that takes image from any size into given size, preserving number of channels
		self.adAvgPool = nn.AdaptiveAvgPool2d(1)
		# After Adaptive Pooling we end up with a matrix, this flattens it into a single vector.
		self.flatten = nn.Flatten()
		# Final classification fully connected layer, taking vector of size number of channels to number of classes
		self.classifier = nn.Linear(64, num_classes)
	
	def forward(self, x):
		# Explanation of size (Batch size, Channels, Heights, Width)
		# x = x  # out: (B, 3, 256, 256)
		x = self.pool(F.relu(self.conv1(x)))  # out: (B, 8, 128, 128)
		x = self.pool(F.relu(self.conv2(x)))  # out: (B, 16, 64, 64)
		x = self.pool(F.relu(self.conv3(x)))  # out: (B, 32, 32, 32)
		x = self.adAvgPool(F.relu(self.conv4(x)))  # out: (B, 64, 1, 1)
		# After extracting hopefully all the schemantic, feature information,
		# we pass the result through full connected classification layer.
		x = self.classifier(self.flatten(x))
		return x  # Return the raw classification values, since SoftMax will be applied in CrossEntropy Loss
