import os
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader

from cnn import CNN
from dataset import PLWordsDataset, transformation

N_EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MOMENTUM = 0.9

DEVICE = "mps"

LOG_INTERVAL = 1
SAVE_STEP = 100


def train(loader, device):
	model = CNN(len(loader)).to(device)
	
	criterion = nn.CrossEntropyLoss()
	optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
	
	print("### Starting training ###")
	for epoch in range(N_EPOCHS):
		start_time = time.time()
		model.train()
		
		running_loss = 0.0
		correct = 0
		
		for images, labels in loader:
			images, labels = images.to(DEVICE), labels.to(DEVICE)
			
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
		

if __name__ == '__main__':
	os.makedirs("steps", exist_ok=True)
	dataset = PLWordsDataset("./data/dataset.csv", transform=transformation)
	data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
	train(data_loader, DEVICE)
