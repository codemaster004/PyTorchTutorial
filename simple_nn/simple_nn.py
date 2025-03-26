import os

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD

from torch.utils.data import Dataset, DataLoader, random_split

TRAIN_VAL_SPLIT = 0.80
BATCH_SIZE = 1
N_EPOCHS = 100
LOG_INTERVAL = 10
LEARNING_RATE = 0.001

DEVICE = "cpu"


class HeartRateDataset(Dataset):
	def __init__(self, file_path, x_cols, y_cols):
		self.data = pd.read_csv(file_path)
		self.data = self.data.astype("float32")
		self.features = self.data[x_cols].values
		self.labels = self.data[y_cols].values.flatten()
		self.labels = self.labels.astype("int64")
	
	def __len__(self):
		return self.data.shape[0]
	
	def __getitem__(self, idx):
		x = torch.tensor(self.features[idx], dtype=torch.float32)
		y = torch.tensor(self.labels[idx], dtype=torch.int64)
		return x, y


class BasicNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.input_layer = nn.Linear(5, 10)
		self.hidden_layer_1 = nn.Linear(10, 10)
		self.output_layer = nn.Linear(10, 2)
	
	def forward(self, x):
		x = torch.relu(self.input_layer(x))
		x = torch.relu(self.hidden_layer_1(x))
		x = self.output_layer(x)
		
		return x


def load_dataset():
	return HeartRateDataset(
		"./data/heart-prediction-dataset.csv",
		["Age", "Gender", "BloodPressure", "Cholesterol", "HeartRate"],
		["HeartDisease"]
	)


def split_dataset(dataset):
	train_size = int(len(dataset) * TRAIN_VAL_SPLIT)
	val_size = len(dataset) - train_size
	
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
	
	return train_dataset, train_loader, val_dataset, val_loader


def train(train_loader, device):
	model = BasicNN()
	model.to(device)
	
	criterion = nn.CrossEntropyLoss()
	optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
	
	for epoch in range(N_EPOCHS):
		model.train()
		running_loss = 0.0
		correct, total = 0, 0
		
		for inputs, labels in train_loader:
			inputs, labels = inputs.to(device), labels.to(device)
			
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item() * inputs.size(0)
			
			_, predicted = torch.max(outputs, 1)
			correct += (predicted == labels).sum().item()
			total += labels.size(0)
		
		epoch_loss = running_loss / total
		epoch_accuracy = correct / total
		
		if (epoch + 1) % LOG_INTERVAL == 0:
			print(f"Epoch [{epoch + 1}/{N_EPOCHS}], loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy * 100:.2f}%")
	
	return model


def validate(model, dataloader, device):
	criterion = nn.CrossEntropyLoss()
	
	model.eval()
	val_loss = 0.0
	correct = 0
	total = 0
	
	# disable gradient calculation
	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs, labels = inputs.to(device), labels.to(device)
			
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			
			val_loss += loss.item()
			
			_, predicted = torch.max(outputs, 1)
			correct += (predicted == labels).sum().item()
			total += labels.size(0)
	
	avg_loss = val_loss / len(dataloader)
	accuracy = 100 * correct / total
	print(f"Average loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%")
	

def main():
	dataset = load_dataset()
	_, train_loader, _, val_loader = split_dataset(dataset)
	
	device = torch.device(DEVICE)
	
	model = train(train_loader, device)
	
	validate(model, val_loader, device)
	
	torch.save(model.state_dict(), 'steps/final_weights.pth')


def loss_example():
	# Define 3 Samples and 4 Classes
	preds = torch.randn(3, 4)
	labels = torch.LongTensor([0, 1, 2])  # label IDs so example [label0, label1, label2]
	
	print(f'Input data: \n{preds}')
	print(f'\nLabels:\n{labels}')
	
	criterion = nn.CrossEntropyLoss()
	loss = criterion(preds, labels)
	
	print(f'Loss: {loss}')


def dataset_example():
	dataset = HeartRateDataset(
		"./data/heart-prediction-dataset.csv", ["BloodPressure", "Cholesterol", "HeartRate"], ["HeartDisease"]
	)
	loader = DataLoader(dataset, batch_size=16, shuffle=True)
	# We enumerate over the whole dataset, in random order, in batches of 16
	for batch_idx, (inputs, targets) in enumerate(loader):
		print(f"Batch {batch_idx}:")
		print(f"  Inputs shape: {inputs.shape}")
		print(f"  Targets shape: {targets.shape}")


if __name__ == '__main__':
	# loss_example()
	# dataset_example()
	
	main()
	
	# Load model
	os.makedirs("steps", exist_ok=True)
	model = torch.load('steps/final_weights.pth')
