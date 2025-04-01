import numpy as np
from pettingzoo.classic import connect_four_v3


def episode(environment, de_omt=1):
	# Reset for safety
	environment.reset()
	
	state_t = np.zeros((6, 7), dtype=np.int8)  # Begging state (board) filled with zeros
	actions = np.array([0, 1, 2, 3, 4, 5, 6])  # all possible actions
	mask = np.ones(7)  # initial mask for actions, all allowed
	
	trajectory_m = []  # (s_0, a_0, r_1), (s_1, a_1, r_2), ...
	at_state_t = False
	while not at_state_t:
		# Picking an action
		action_t = actions[mask.astype(bool)]
		action_t = np.random.choice(action_t)
		# Perform the action
		environment.step(action_t)
		# Getting new state and reward from the environment
		observation, reward_t, termination, _, _ = environment.last()
		if termination:
			at_state_t = True
		# Update trajectory
		trajectory_m.append((state_t, action_t, reward_t * -1))  # * -1 since it is reward for next (incorrect) player
		# Update state and action mask
		state_t, mask = observation['observation'], observation['action_mask']
		
		# Reformatting the state
		state_t[:, :, 1] *= -1
		state_t = state_t.sum(axis=2)
	
	trajectory_m_p0 = [trajectory_m[i] for i in range(0, len(trajectory_m), 2)]
	trajectory_m_p1 = [trajectory_m[i] for i in range(1, len(trajectory_m), 2)]
	
	return trajectory_m_p0, trajectory_m_p1


if __name__ == '__main__':
	env = connect_four_v3.env()
	env.reset()
	
	trajectory_p0, trajectory_p1 = episode(env)
	# for i in range(100_000):
	# 	trajectory = generate_trajectory(env)
	# 	if i % 1000 == 0:
	# 		print(f"Iteration {i}")
	
	print(trajectory_p0[0])
	print(trajectory_p0[1])
	print(trajectory_p0[-1])
	print()
	print(trajectory_p1[0])
	print(trajectory_p1[1])
	print(trajectory_p1[-1])
