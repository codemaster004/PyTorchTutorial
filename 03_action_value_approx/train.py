import pandas as pd
import logging
from pettingzoo.classic import connect_four_v3

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from enviroment import episode
from nn import LinearNN
from dataset import RLDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/rl_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate_policy(model):
	env = connect_four_v3.env()
	env.reset()
	
	trajectories = []
	for i in range(1_000):
		trajectory_p0, trajectory_p1 = episode(env, model, lam=0.1, epsilon=0.5)
		trajectories.extend(trajectory_p0)
		trajectories.extend(trajectory_p1)
	
	return pd.DataFrame(trajectories)
	

def train():
	model = LinearNN(43, 1)
	
	criterion = nn.MSELoss()
	optimizer = SGD(model.parameters(), lr=0.001)
	
	logger.info("Starting DeepMonteCarlo training")
	for epoch in range(30):
		model.eval()
		df = evaluate_policy(model)
		dataset = RLDataset(df)
		dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
		
		total_loss = 0.0
		
		model.train()
		for state_action, reward in dataloader:
			state_action, reward = state_action.to("cpu"), reward.to("cpu")
			
			y = model(state_action)
			loss = criterion(y, reward)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item() * reward.size(0)
		
		epoch_loss = total_loss / len(dataset)
		if (epoch+1) % 10 == 0:
			logger.info(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
	
	return model


if __name__ == '__main__':
	trained_model = train()
	torch.save(trained_model.state_dict(), 'steps/final_weights.pth')
