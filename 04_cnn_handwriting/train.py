import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from cnn import CNN
from dataset import PLWordsDataset, transformation

N_EPOCHS = 500
BATCH_SIZE = 48
LEARNING_RATE = 1e-3
MOMENTUM = 0.0

DEVICE = "mps"

LOG_INTERVAL = 1
SAVE_STEP = 20


def train(model, loader, device):
	criterion = nn.CrossEntropyLoss()
	optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
	
	print("### Starting training ###")
	for epoch in range(N_EPOCHS):
		start_time = time.time()
		model.train()
		
		running_loss = 0.0
		correct = 0
		
		for images, labels in loader:
			images, labels = images.to(device), labels.to(device)
			
			outputs = model(images)
			loss = criterion(outputs, labels)
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item() * labels.size(0)
			
			_, predicted = torch.max(outputs, 1)
			correct += predicted.eq(labels).sum().item()
		
		epoch_loss = running_loss / len(loader.dataset)
		epoch_accuracy = correct / len(loader.dataset)
		
		if (epoch + 1) % LOG_INTERVAL == 0:
			end_time = time.time()
			duration = end_time - start_time
			print(f"Epoch [{epoch + 1}/{N_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.2f}% ", end="")
			print(f"Speed: {str(round(duration / LOG_INTERVAL, 1)) + ' s/it' if duration / LOG_INTERVAL > 1.0 else str(round(LOG_INTERVAL / duration)) + ' it/s'}")
		if epoch % SAVE_STEP == 0:
			torch.save(model.state_dict(), f'steps/{str(epoch)}.pth')


def split():
	df = pd.read_csv("./data/dataset.csv")
	train_df, val_df = train_test_split(df, test_size=0.4, random_state=42, shuffle=True)
	train_df.to_csv("./data/train.csv", index=False)
	val_df.to_csv("./data/val.csv", index=False)
	

if __name__ == '__main__':
	os.makedirs("steps", exist_ok=True)
	split()
	dataset = PLWordsDataset("./data/train.csv", transform=transformation)
	data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
	
	model = CNN(len(dataset)).to(DEVICE)
	model.load_state_dict(torch.load('steps/40.pth'))
	train(model, data_loader, DEVICE)
