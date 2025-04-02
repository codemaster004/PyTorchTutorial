import os

from pettingzoo.classic import connect_four_v3  # make sure to run `pip install 'pettingzoo[classic]'`
import pandas as pd
import logging
import yaml
import time

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from enviroment import episode
from nn import LinearNN
from dataset import RLDataset

# More logging capabilities than print()
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


# Function for reading config file, making sure everything is there
def read_config(config_path):
	with open(config_path, "r") as yaml_f:
		config = yaml.safe_load(yaml_f)
	# Check if subgroups are there
	if 'train' not in config:
		raise ValueError("Config file must contain 'train' section")
	if 'network' not in config:
		raise ValueError("Config file must contain 'network' section")
	
	# Check if all options are in the config
	if any([opt not in config['train'] for opt in [
		'learning_rate', 'batch_size', 'epochs', 'episodes', 'device', 'discounting', 'epsilon', 'log_step', 'save_step', 'momentum']]):
		raise ValueError("Missing one of required options")
	if any([opt not in config['network'] for opt in ['input_dim', 'output_dim', 'hidden_dim']]):
		raise ValueError("Missing one of required options")
	# return sub configs
	return config['train'], config['network']


def evaluate_policy(model, episodes, lam, epsilon):
	env = connect_four_v3.env()  # Initiate environment
	env.reset()  # Start by resetting
	
	trajectories = []  # Raw trajectories from each game joined together [[state vec, action index, reward], ...]
	for i in range(episodes):
		# Run episode with discounting parameter lambda and random move chance epsilon
		trajectory_p0, trajectory_p1 = episode(env, model, lam=lam, epsilon=epsilon)
		trajectories.extend(trajectory_p0)  # Add trajectory of player 0
		trajectories.extend(trajectory_p1)  # Add trajectory of player 1
	
	return pd.DataFrame(trajectories)


def log_train_info(loss, epochs, epoch, interval, start_time, end_time, total_time):
	duration = end_time - start_time
	# Determine unit of time and iteration
	if duration / interval > 1.0:
		info = str(round(duration / interval, 1)) + ' s/it'
	else:
		info = str(round(interval / duration)) + ' it/s'
	# Calculation for remaining time
	left_epochs = epochs - epoch
	eta = str(round(left_epochs * (total_time / (epoch + 1)), 1))
	
	logger.info(f"Epoch {epoch + 1:05d} Loss: {loss:.4f} ETA: [ {info} | {eta}s ] ")


def train():
	train_opt, network_opt = read_config("config.yml")  # Read options form config
	
	model = LinearNN(network_opt['input_dim'], network_opt['output_dim'], network_opt['hidden_dim'])
	
	criterion = nn.MSELoss()
	optimizer = SGD(model.parameters(), lr=train_opt['learning_rate'], momentum=train_opt['momentum'])
	
	# A LOT of printing
	logger.info("### Starting DeepMonteCarlo training")
	logger.info("### Training with:")
	logger.info(f"###    Device: {train_opt['device']}, Batch size: {train_opt['batch_size']}")
	logger.info(f"###    Learning rate: {train_opt['learning_rate']}, Momentum: {train_opt['momentum']}")
	logger.info("### Simulation Settings:")
	logger.info(f"###    Episodes: {train_opt['episodes']}")
	logger.info(f"###    Discounting: {train_opt['discounting']}, Epsilon: {train_opt['epsilon']}")
	
	start_time = time.time()  # Start counting time
	total_time = 0  # Total time of training
	for epoch in range(train_opt['epochs']):
		model.eval()  # Set the model into evaluation mode
		
		df = evaluate_policy(model, train_opt['episodes'], train_opt['discounting'], train_opt['epsilon'])  # Run games
		# Convert collected games states into a dataset
		dataset = RLDataset(df)
		dataloader = DataLoader(dataset, batch_size=train_opt['batch_size'], shuffle=True)
		
		model.train()  # Set the model into training mode
		
		total_loss = 0.0
		for state_action, reward in dataloader:
			state_action, reward = state_action.to("cpu"), reward.to("cpu")
			# Standard loss optimization
			y = model(state_action)
			loss = criterion(y, reward)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# Remember the loss for logs
			total_loss += loss.item() * reward.size(0)
		
		epoch_loss = total_loss / len(dataset)
		
		# Logging time and stuff
		if (epoch + 1) % train_opt['log_step'] == 0:
			end_time = time.time()
			total_time += end_time - start_time
			log_train_info(epoch_loss, train_opt['epochs'], epoch, train_opt['log_step'],
			               start_time, end_time, total_time)
			start_time = time.time()
		# Save model in middle steps
		if (epoch + 1) % train_opt['save_step'] == 0:
			torch.save(model.state_dict(), f"./steps/{epoch:05d}.pth")
	
	return model


if __name__ == '__main__':
	os.makedirs('steps', exist_ok=True)
	os.makedirs('steps', exist_ok=True)
	
	trained_model = train()
	torch.save(trained_model.state_dict(), './steps/final_weights.pth')
