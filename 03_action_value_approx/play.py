import random

import numpy as np
from pettingzoo.classic import connect_four_v3

import torch

from nn import LinearNN
from enviroment import eval_best_action


def main(model):
	# Init Environment
	env = connect_four_v3.env(render_mode="human")
	env.reset()
	# Players, and choose random
	players = ["human", "computer"]
	player = random.choice(players)
	# All actions that could be taken
	actions = np.array([0, 1, 2, 3, 4, 5, 6])
	
	while True:
		observation, reward, termination, truncation, info = env.last()
		# End condition
		if termination or truncation:
			break
		# Reading state, and mask for allowed actions
		state_t, mask = observation['observation'], observation['action_mask']
		
		state_t[:, :, 1] *= -1  # multiply opponent player paws to be represented by `-1`
		state_t = state_t.sum(axis=2)  # Sum both dimension, for both players to be on a single matrix
		state_t = state_t.flatten()  # Convert the matrix into a single vector
		
		# Players action handling
		if player == "human":
			action = int(input("Pick action: "))
		else:
			action = actions[mask.astype(bool)]
			action = eval_best_action(state_t, action, model)
		
		env.step(action)
		# Change players
		if player == "human":
			player = "computer"
		else:
			player = "human"
	env.close()


if __name__ == '__main__':
	model = LinearNN(49, 1, 20)
	model.load_state_dict(torch.load('steps/final_weights.pth'))
	model.eval()
	main(model)
