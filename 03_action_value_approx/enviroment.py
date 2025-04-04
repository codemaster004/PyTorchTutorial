from pettingzoo.classic import connect_four_v3
import numpy as np

import torch


def calc_cumsum_rewards(trajectory_m, lam):
	# todo: state of draw
	if trajectory_m[-1][2] != 1:
		trajectory_m[-1][2] = lam * -1
	
	# Calculating rewards from final reward with discounting lam
	running_sum = 0.0
	for t in reversed(range(len(trajectory_m))):
		running_sum = trajectory_m[t][2] + lam * running_sum
		trajectory_m[t][2] = running_sum
	return trajectory_m  # return trajectory with updated rewards


def eval_best_action(state_t, actions, model):
	action_t, max_reward = actions[0], np.float32("-inf")  # some defaults
	# To disable gradient calculation
	with torch.no_grad():
		for action in actions:
			action_one_hot = np.zeros(7)
			action_one_hot[action] = 1
			x = torch.tensor(np.append(state_t, action_one_hot), dtype=torch.float32)
			y = model(x)
			if y > max_reward:
				action_t, max_reward = action, y
	# return action yielding most reward
	return action_t


def episode(environment, model, lam=1.0, epsilon=1.0):
	# Reset for safety
	environment.reset()
	
	state_t = np.zeros((42,), dtype=np.int8)  # Begging state (board) filled with zeros
	actions = np.array([0, 1, 2, 3, 4, 5, 6])  # all possible actions
	mask = np.ones(7)  # initial mask for actions, all allowed
	
	# todo: maybe not include first few moves, or not allways (stabilising the dataset)
	trajectory_m = []  # [s_0, a_0, r_1], [s_1, a_1, r_2], ...
	at_state_T = False
	while not at_state_T:
		# Picking an action
		action_t = actions[mask.astype(bool)]
		if np.random.rand() < epsilon:
			action_t = np.random.choice(action_t)
		else:
			action_t = eval_best_action(state_t, action_t, model)
		
		# Perform the action
		environment.step(action_t)
		# Getting new state and reward from the environment
		observation, reward_t, termination, _, _ = environment.last()
		if termination:
			at_state_T = True
		# Update trajectory
		trajectory_m.append([state_t, action_t, reward_t * -1])  # * -1 since it is reward for next (incorrect) player
		# Update state and action mask
		state_t, mask = observation['observation'], observation['action_mask']
		
		# Reformatting the state
		state_t[:, :, 1] *= -1
		state_t = state_t.sum(axis=2)
		state_t = state_t.flatten()
	
	# Save trajectories with respect to players, with calculated cumulative rewards
	trajectory_m_p0 = [trajectory_m[i] for i in range(0, len(trajectory_m), 2)]
	trajectory_m_p0 = calc_cumsum_rewards(trajectory_m_p0, lam)
	trajectory_m_p1 = [trajectory_m[i] for i in range(1, len(trajectory_m), 2)]
	trajectory_m_p1 = calc_cumsum_rewards(trajectory_m_p1, lam)

	return trajectory_m_p0, trajectory_m_p1


if __name__ == '__main__':
	env = connect_four_v3.env()
	env.reset()
	
	trajectory_p0, trajectory_p1 = episode(env, None, lam=0.9, epsilon=0)
	
	# for s, a, r in trajectory_p0:
	# 	print(s, a, r)
