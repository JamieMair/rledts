import sys
import os
import math
import numpy as np
import numpy.random
from scipy import linalg
import approximations

class ring_dynamics_fourier(object):

	def __init__(self, parameters):
		self.model = parameters['model']
		self.average_reward = 0
		self.entropy = 0
		self.current = 0
		self.reward_learning_rate = parameters['reward_learning_rate']
		self.value_approximation = approximations.fourier_basis(
			parameters['weight_number'], self.model.ring_length)
		self.value_learning_rate = parameters['value_learning_rate']
		self.potential_approximation = approximations.fourier_basis(
			parameters['weight_number'], self.model.ring_length)
		self.dynamic_learning_rate = parameters['dynamic_learning_rate']
		self.time = 0
		self.current_features = np.zeros((parameters['weight_number']))

	def train(self, steps):
		average_reward_vs_time = np.zeros((steps + 1))
		position_vs_time = np.zeros((steps + 1))
		position_vs_time[0] = self.model.current_state
		self.current_features = self.value_approximation.features(
			self.model.current_state)
		while self.time < steps:
			transition_probability, transition_gradient = self._transition()
			self._update(transition_probability, transition_gradient)
			self.model.current_state = self.model.next_state
			self.time += 1
			average_reward_vs_time[self.time] = self.average_reward
			position_vs_time[self.time] = self.model.current_state
		self.time = 0
		return average_reward_vs_time, position_vs_time

	def _transition(self):
		potential_gradient = self.potential_approximation.features(
			self.model.current_state)
		potential = self.potential_approximation.weights @ potential_gradient
		exponentiated_potential = math.exp(potential)
		right_probability = exponentiated_potential / (exponentiated_potential + 1)
		random = numpy.random.random()
		if random < right_probability:
			self.model.next_state = (self.model.current_state+1) % self.model.ring_length
			transition_gradient = (1 - right_probability) * potential_gradient
			return right_probability, transition_gradient
		else:
			self.model.next_state = (self.model.current_state-1) % self.model.ring_length
			transition_gradient = -right_probability * potential_gradient
			return 1 - right_probability, transition_gradient

	def _update(self, transition_probability, transition_gradient):
		reward = self.model.reward(transition_probability)
		next_features = self.value_approximation.features(self.model.next_state)
		current_value = self.current_features @ self.value_approximation.weights
		next_value = next_features @ self.value_approximation.weights
		td_error = (next_value + reward - self.average_reward - current_value)
		self.average_reward += self.reward_learning_rate * td_error
		#elf.average_reward += self.reward_learning_rate * (reward - self.average_reward)
		self.entropy += self.reward_learning_rate*0.1 * (-math.log(transition_probability) 
													- self.entropy)
		current = ((self.model.next_state-1)%self.model.ring_length 
				   == self.model.current_state)
		self.current += self.reward_learning_rate*0.1 * (current - self.current)
		self.value_approximation.weights += (self.value_learning_rate * td_error
											 * self.current_features)
		self.potential_approximation.weights += (self.dynamic_learning_rate * td_error 
												 * transition_gradient)
		self.current_features = next_features

	def potential(self):
		potential = np.zeros((self.model.ring_length))
		for state in range(self.model.ring_length):
			potential_gradient = self.potential_approximation.features(state)
			potential[state] = self.potential_approximation.weights @ potential_gradient
		return potential

	def dynamics(self):
		probabilities = np.zeros((2, self.model.ring_length))
		for state in range(self.model.ring_length):
			potential_gradient = self.potential_approximation.features(state)
			potential = self.potential_approximation.weights @ potential_gradient
			left_probability = 1/(math.exp(potential) + 1)
			probabilities[1][state] = 1 - left_probability
			probabilities[0][state] = left_probability
		return probabilities

	def values(self):
		values = np.zeros((self.model.ring_length))
		for state in range(self.model.ring_length):
			values[state] = (self.value_approximation.features(state) 
							 @ self.value_approximation.weights)
		return values

	def stationary_state(self):
		probabilities = self.dynamics()
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


class ring_dynamics_fourier_discounted(object):

	def __init__(self, parameters):
		self.model = parameters['model']
		self.average_reward = 0
		self.entropy = 0
		self.current = 0
		self.reward_learning_rate = parameters['reward_learning_rate']
		self.discount = parameters['discount']
		self.value_approximation = approximations.fourier_basis(
			parameters['weight_number'], self.model.ring_length)
		self.value_learning_rate = parameters['value_learning_rate']
		print(self.value_approximation.weights)
		self.potential_approximation = approximations.fourier_basis(
			parameters['weight_number'], self.model.ring_length)
		self.dynamic_learning_rate = parameters['dynamic_learning_rate']
		self.time = 0
		self.current_features = np.zeros((parameters['weight_number']))

	def train(self, steps):
		average_reward_vs_time = np.zeros((steps + 1))
		position_vs_time = np.zeros((steps + 1))
		position_vs_time[0] = self.model.current_state
		self.current_features = self.value_approximation.features(
			self.model.current_state)
		while self.time < steps:
			transition_probability, transition_gradient = self._transition()
			self._update(transition_probability, transition_gradient)
			self.model.current_state = self.model.next_state
			self.time += 1
			average_reward_vs_time[self.time] = self.average_reward
			position_vs_time[self.time] = self.model.current_state
		self.time = 0
		return average_reward_vs_time, position_vs_time

	def _transition(self):
		potential_gradient = self.potential_approximation.features(
			self.model.current_state)
		potential = self.potential_approximation.weights @ potential_gradient
		exponentiated_potential = math.exp(potential)
		right_probability = exponentiated_potential / (exponentiated_potential + 1)
		random = numpy.random.random()
		if random < right_probability:
			self.model.next_state = (self.model.current_state+1) % self.model.ring_length
			transition_gradient = (1 - right_probability) * potential_gradient
			return right_probability, transition_gradient
		else:
			self.model.next_state = (self.model.current_state-1) % self.model.ring_length
			transition_gradient = -right_probability * potential_gradient
			return 1 - right_probability, transition_gradient

	def _update(self, transition_probability, transition_gradient):
		reward = self.model.reward(transition_probability)
		next_features = self.value_approximation.features(self.model.next_state)
		current_value = self.current_features @ self.value_approximation.weights
		next_value = next_features @ self.value_approximation.weights
		td_error = (self.discount*next_value + reward - current_value)
		self.average_reward += self.reward_learning_rate * (reward - self.average_reward)
		self.entropy += self.reward_learning_rate*0.1 * (-math.log(transition_probability) 
													- self.entropy)
		current = ((self.model.next_state-1)%self.model.ring_length 
				   == self.model.current_state)
		self.current += self.reward_learning_rate*0.1 * (current - self.current)
		self.value_approximation.weights += (self.value_learning_rate * td_error
											 * self.current_features)
		self.potential_approximation.weights += (self.dynamic_learning_rate * td_error 
												 * transition_gradient)
		self.current_features = next_features

	def potential(self):
		potential = np.zeros((self.model.ring_length))
		for state in range(self.model.ring_length):
			potential_gradient = self.potential_approximation.features(state)
			potential[state] = self.potential_approximation.weights @ potential_gradient
		return potential

	def dynamics(self):
		probabilities = np.zeros((2, self.model.ring_length))
		for state in range(self.model.ring_length):
			potential_gradient = self.potential_approximation.features(state)
			potential = self.potential_approximation.weights @ potential_gradient
			left_probability = 1/(math.exp(potential) + 1)
			probabilities[1][state] = 1 - left_probability
			probabilities[0][state] = left_probability
		return probabilities

	def values(self):
		values = np.zeros((self.model.ring_length))
		for state in range(self.model.ring_length):
			values[state] = (self.value_approximation.features(state) 
							 @ self.value_approximation.weights)
		return values

	def stationary_state(self):
		probabilities = self.dynamics()
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