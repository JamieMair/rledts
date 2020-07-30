import sys
import os
import math
import numpy as np
import numpy.random
from scipy import linalg

class ring_dynamics_tabular(object):

	def __init__(self, parameters):
		self.model = parameters['model']
		if 'initial_reward' in parameters:
			self.average_reward = parameters['initial_reward']
		else:
			self.average_reward = 0
		if 'initial_entropy' in parameters:
			self.entropy = parameters['initial_entropy']
		else:
			self.entropy = 0
		if 'initial_current' in parameters:
			self.current = parameters['initial_current']
		else:
			self.current = 0
		self.reward_learning_rate = parameters['reward_learning_rate']
		self.values = np.zeros((self.model.ring_length))
		self.value_learning_rate = parameters['value_learning_rate']
		self.dynamic_weights = np.zeros((self.model.ring_length, 2))
		self.dynamic_learning_rate = parameters['dynamic_learning_rate']
		self.time = 0

	def initialize(self, steps):
		while self.time < steps:
			transition_probability, transition = self._transition()
			self._initialization_update(transition_probability, transition)
			self.model.current_state = self.model.next_state
			self.time += 1
		self.time = 0
		return

	def _initialization_update(self, transition_probability, transition):
		reward = self.model.reward(transition_probability)
		td_error = (self.values[self.model.next_state] + reward - self.average_reward
					- self.values[self.model.current_state])
		self.average_reward += (reward - self.average_reward)/(self.time + 1)
		self.entropy += (-math.log(transition_probability) - self.entropy)/(self.time + 1)
		self.current += (0.5*(self.model.next_state-self.model.current_state+1)
							 - self.current)/(self.time + 1)
		self.values[self.model.current_state] += td_error/(self.time + 1)

	def train(self, steps, output = True):
		if output == True:
			average_reward = np.zeros((steps + 1))
			average_reward[0] = self.average_reward
			entropy = np.zeros((steps + 1))
			entropy[0] = self.entropy
			current = np.zeros((steps + 1))
			current[0] = self.current
			position = np.zeros((steps + 1))
			position[0] = self.model.current_state
			while self.time < steps:
				transition_probability, transition = self._transition()
				self._update(transition_probability, transition)
				self.model.current_state = self.model.next_state
				self.time += 1
				average_reward[self.time] = self.average_reward
				entropy[self.time] = self.entropy
				current[self.time] = self.current
				position[self.time] = self.model.current_state
			self.time = 0
			return average_reward, entropy, current, position
		else:
			while self.time < steps:
				transition_probability, transition = self._transition()
				self._update(transition_probability, transition)
				self.model.current_state = self.model.next_state
				self.time += 1
			self.time = 0
			return

	def _transition(self):
		norm = (np.exp(self.dynamic_weights[self.model.current_state][1]) 
				+ np.exp(self.dynamic_weights[self.model.current_state][0]))
		right_probability = np.exp(self.dynamic_weights[self.model.current_state][1])/norm
		random = numpy.random.random()
		if random < right_probability:
			self.model.next_state = (self.model.current_state+1) % self.model.ring_length
			return right_probability, 1
		else:
			self.model.next_state = (self.model.current_state-1) % self.model.ring_length
			return 1 - right_probability, 0

	def _update(self, transition_probability, transition):
		reward = self.model.reward(transition_probability)
		td_error = (self.values[self.model.next_state] + reward - self.average_reward
					- self.values[self.model.current_state])
		self.average_reward += self.reward_learning_rate * td_error
		self.entropy += self.reward_learning_rate*0.1 * (-math.log(transition_probability) 
													- self.entropy)
		current = ((self.model.next_state-1)%self.model.ring_length 
				   == self.model.current_state)
		self.current += self.reward_learning_rate*0.1 * (current - self.current)
		self.values[self.model.current_state] += self.value_learning_rate * td_error
		policy_update = self.dynamic_learning_rate * td_error * (1-transition_probability)
		self.dynamic_weights[self.model.current_state][transition] += policy_update
		self.dynamic_weights[self.model.current_state][1-transition] -= policy_update

	def dynamics(self):
		exp_potentials = np.exp(self.dynamic_weights)
		return exp_potentials/np.sum(exp_potentials, axis = 1)[:, np.newaxis]

	def potential(self):
		return self.dynamic_weights[:,1] - self.dynamic_weights[:,0]

	def stationary_state(self):
		probabilities = self.dynamics().T
		master_operator = np.zeros((self.model.ring_length, self.model.ring_length), dtype = np.complex64)
		for position in range(self.model.ring_length):
			master_operator[(position + 1)%self.model.ring_length][position] = (
				probabilities[1][position])
			master_operator[(position - 1)%self.model.ring_length][position] = (
				probabilities[0][position])
		spectrum = linalg.eig(master_operator, left = True)
		sorted_index = spectrum[0].argsort()[::-1]
		eigenvalues = spectrum[0][sorted_index]
		left_eigenvectors = spectrum[1][:,sorted_index]
		right_eigenvectors = spectrum[2][:,sorted_index]
		norm = np.sum(right_eigenvectors[:,0])
		return right_eigenvectors[:,0]/norm
