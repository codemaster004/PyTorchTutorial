import os

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD

from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


TRAIN_VAL_SPLIT = 0.80  # How much data should be used for training, how much for validation
LOG_INTERVAL = 10  # Every how many epochs we log anything

# Why those values? Because it worked good enough
BATCH_SIZE = 32  # Into how many and how large batches, we divide the dataset into
N_EPOCHS = 80  # How many times we train the model (on whole dataset)
LEARNING_RATE = 0.001  # How large portion of the gradient is added to the weights

DEVICE = "cpu"  # Where to perform the calculation, use "cuda" for GPU acceleration


# For optional resource utilisation we create a custom Dataset class based on torch.utils.data.Dataset
class HeartRateDataset(Dataset):
	def __init__(self, file_path, x_cols, y_cols):
		self.data = pd.read_csv(file_path)  # Load the heart rate data from file
		# Transform entire dataset into float32
		self.data = self.data[x_cols + y_cols].astype("float32")
		# Store features columns, label columns separately
		self.features = self.data[x_cols].values
		# For CrossEntropy classification loss, single K-classes vector is required
		self.labels = self.data[y_cols].values.flatten()  # Flatten into a single K dim vector
		self.labels = self.labels.astype("int64")  # Required dtype for CrossEntropy is LongInt
	
	def __len__(self):
		return self.data.shape[0]  # Shape: [N_rows, M_features]
	
	def __getitem__(self, idx):
		# Convert here into tensors, more time-consuming, but less resource heavy,
		# than storing entire dataset in tensor form in memory
		x = torch.tensor(self.features[idx], dtype=torch.float32)  # Features
		y = torch.tensor(self.labels[idx], dtype=torch.int64)  # Labels
		return x, y


# Custom small NeuralNetwork, implementation based on torch.nn.Module
class BasicNN(nn.Module):
	def __init__(self):
		super().__init__()  # Not sure if required, but good practise
		# For classification of has Heart Disease we have 5 input features and 2 output classes (does not have, has)
		# Our network will have (why? because random, and it worked ish):
		#   input layer with 5 neurons
		#   fully connected hidden layer with 10 neurons
		#   fully connected hidden layer with 10 neurons
		#   fully connected output layer with 2 neurons
		
		# nn.Linear: y = xA^T + b
		self.input_layer = nn.Linear(5, 10)  # Define InputLayer(5) -> HiddenLayer1(10) weights
		self.hidden_layer = nn.Linear(10, 10)  # Define HiddenLayer1(10) -> HiddenLayer2(10) weights
		self.output_layer = nn.Linear(10, 2)  # Define HiddenLayer2(10) -> OutputLayer(2) weights
	
	def forward(self, x):
		# In the __init__ function, we only defined the weight matrices, but without any information about its order of firing
		# In the forward process function we define what calculations happen, one after, the other.
		# x = x  # input x into the network
		x = torch.relu(self.input_layer(x))  # x = (relu(y) <- y = xW1^T + b1)
		x = torch.relu(self.hidden_layer(x))  # x = (relu(y) <- y = xW2^T + b2)
		x = torch.softmax(self.output_layer(x), dim=1)  # x = (softmax(y) <- y = xW3^T + b3)
		
		return x


# Um very complicated function, I know.
def load_dataset():
	return HeartRateDataset(
		"./data/heart-prediction-dataset.csv",
		["Age", "Gender", "BloodPressure", "Cholesterol", "HeartRate"],
		["HeartDisease"]
	)


# It is a good practice to split the entire dataset into two group one for training, one for validation.
# Validation/testing is a process of checking how much data was classified correctly after training.
# It should be done on unseen data to truly test if model learns correctly.
def split_dataset(dataset):
	train_size = int(len(dataset) * TRAIN_VAL_SPLIT)
	val_size = len(dataset) - train_size
	
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # create loader for training
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # create loader for validation
	
	return train_dataset, train_loader, val_dataset, val_loader


def train(train_loader, device):
	model = BasicNN()  # Create instance of the NN
	model.to(device)  # Convert to picked device
	
	criterion = nn.CrossEntropyLoss()  # Define the los function
	optimizer = SGD(model.parameters(), lr=LEARNING_RATE)  # What method to use when changing weight matrices
	
	for epoch in range(N_EPOCHS):
		model.train()
		running_loss = 0.0  # Total loss cumulated during single epoch
		correct, total = 0, 0
		
		# Iterate over whole dataset in batches
		for inputs, labels in train_loader:
			# convert input and labels to device
			inputs, labels = inputs.to(device), labels.to(device)
			
			outputs = model(inputs)  # calculate the output from the model
			loss = criterion(outputs, labels)  # Calc the loss, difference of calculated output and real values
			# clear the gradients form the previous optimisation pass
			optimizer.zero_grad()
			loss.backward()  # prepare values to adjust the weights by, with respect to the loss
			optimizer.step()  # adjust the weights
			
			# total_loss = loss_value * n_input_examples
			running_loss += loss.item() * inputs.size(0)
			
			_, predicted = torch.max(outputs, 1)  # get the predicted label, index of the most probable label
			correct += (predicted == labels).sum().item()  # count number of correctly predicted labels
			total += labels.size(0)  # Add to the total count of examples, N examples in current batch
		
		epoch_loss = running_loss / total  # avg_epoch_loss = total_epoch_loss / N_examples_in_this_epoch
		epoch_accuracy = correct / total  # epoch_accuracy = N_correctly_classified / N_examples_in_this_epoch
		
		if (epoch + 1) % LOG_INTERVAL == 0:
			print(f"Epoch [{epoch + 1}/{N_EPOCHS}], loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy * 100:.2f}%")
	
	return model


def validate(model, dataloader, device):
	criterion = nn.CrossEntropyLoss()
	
	model.eval()
	val_loss = 0.0
	correct = 0
	total = 0
	
	all_preds = []
	all_labels = []
	
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
			
			all_preds.extend(predicted.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())
	
	avg_loss = val_loss / len(dataloader)
	accuracy = 100 * correct / total
	print(f"Average loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%")
	
	# Compute and display confusion matrix
	cm = confusion_matrix(all_labels, all_preds)
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title('Confusion Matrix')
	plt.show()
	

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
	# os.makedirs("steps", exist_ok=True)
	# model = torch.load('steps/final_weights.pth')
