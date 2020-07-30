import math
import numpy as np

class environment(object):
	"""A simple environment which provides rewards based on excursions."""

	def __init__(self, parameters):
		self.trajectory_length = parameters['trajectory_length']
		self.positivity_bias = parameters['positivity_bias']
		self.target_bias = parameters['target_bias']
		self.action = 0
		self.state = [0, 0]
		self.terminal_state = False

	def _reward(self):
		"""Calculates the reward for the last transition to occur."""
		if self.state[0] < 0:
			reward = -self.positivity_bias# * abs(self.state[0])
		else:
			reward = 0
		if self.state[1] == self.trajectory_length:
			reward -= self.target_bias * abs(self.state[0])
		return reward

	def step(self, action):
		"""Updates the environment state based on the input action."""
		self.action = action
		self.state[0] += 2*action - 1
		self.state[1] += 1
		if self.state[1] == self.trajectory_length:
			self.terminal_state = True
		return self.state, self._reward(), self.terminal_state, None

	def kl_regularization(self, state, action, action_probability):
		return -math.log(action_probability/0.5)

	def reset(self):
		"""Resets the environment state and terminal boolean."""
		self.action = 0
		self.state = [0, 0]
		self.terminal_state = False
		return self.state

class analysis(object):
	
	def __init__(self, success_learning_rate, entropy_learning_rate):
		self.successes = []
		self.average_success = 0
		self.success_learning_rate = success_learning_rate
		self.success = 1
		self.entropies = []
		self.average_entropy = 0
		self.entropy_learning_rate = entropy_learning_rate
		self.entropy = 0

	def per_step(self, past_state, action, current_state, action_probability):
		if current_state[0] < 0:
			self.success = 0
		self.entropy -= math.log(action_probability)

	def reset(self):
		self.success = 1
		self.entropy = 0
	
	def per_episode(self, state):
		if state[0] != 0:
			self.success = 0
		self.average_success += self.success_learning_rate*(self.success 
															- self.average_success)
		self.average_entropy += self.entropy_learning_rate*(self.entropy
															- self.average_entropy)
		self.reset()
		self.successes.append(self.average_success)
		self.entropies.append(self.average_entropy)

	def evaluation_reset(self):
		self.average_success = 0
		self.average_entropy = 0

	def per_sample(self, state, sample):
		if state[0] != 0:
			self.success = 0
		self.average_success += (self.success - self.average_success)/sample
		self.average_entropy += (self.entropy - self.average_entropy)/sample
		self.reset()

class gauge(object):

	def __init__(self, parameters):
		self.trajectory_length = parameters['trajectory_length']
		self.positivity_bias = parameters['positivity_bias']
		self.target_bias = parameters['target_bias']
		self._construct_gauge()

	def _bias(self, current_state):
		if current_state[0] < self.trajectory_length:
			reward = -self.positivity_bias * abs(current_state[0] 
												 - self.trajectory_length)
		else:
			reward = 0
		if current_state[1] == self.trajectory_length:
			reward -= self.target_bias * abs(current_state[0] - self.trajectory_length)
		return math.exp(reward)

	def _construct_gauge(self):
		self.gauge = np.zeros((self.trajectory_length*2 + 1, 
							   self.trajectory_length + 1),
							  dtype = np.float32)
		self.gauge.T[-1] = 1
		for index in range(self.trajectory_length):
			time = self.trajectory_length - index
			self.gauge[1][time - 1] += (0.5 * self.gauge[0][time]
				* self._bias([0,time]))
			for state in range(1, self.trajectory_length*2):
				self.gauge[state + 1][time - 1] += (0.5 * self.gauge[state][time]
					* self._bias([state,time]))
				self.gauge[state - 1][time - 1] += (0.5 * self.gauge[state][time]
					* self._bias([state,time]))
			self.gauge[self.trajectory_length*2 - 1][time - 1] += (
				0.5 * self.gauge[self.trajectory_length*2][time]
				* self._bias([self.trajectory_length*2,time]))
		self.max_return = math.log(self.gauge[self.trajectory_length][0])
		self.max_ent_return = self.max_return + self.trajectory_length*np.log(2)
		return self.gauge

	def update_model(self, new_parameters):
		self.trajectory_length = new_parameters['trajectory_length']
		self.positivity_bias = new_parameters['positivity_bias']
		self.target_bias = new_parameters['target_bias']
		self._construct_gauge()

def state_probabilities(up_probabilities, trajectory_length):
		trajectory_length = trajectory_length
		probabilities = np.zeros((trajectory_length*2 + 1, trajectory_length + 1),
								 dtype = np.float32)
		probabilities[0][0] = 1
		for time in range(trajectory_length):
			for state in range(trajectory_length*2 + 1):
				probabilities[(state + 1)%(trajectory_length*2 + 1)][time + 1] += (
					probabilities[state][time] * up_probabilities[state][time])
				probabilities[state - 1][time + 1] += (probabilities[state][time]
													  * (1-up_probabilities[state][time]))
		probabilities = np.concatenate(
			(probabilities[trajectory_length+1:
						   trajectory_length*2+1,:],
			 probabilities[0:trajectory_length+1,:]), axis = 0)
		return probabilities