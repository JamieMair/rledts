import sys
import os
import numpy as np
import numpy.random

class ring_model_base(object):

	def __init__(self, parameters):
		self.ring_length = parameters['ring_length']
		self.bias = parameters['bias']
		self.current_state = 0
		self.next_state = 0

	def reward(self, transition_probability):
		bias_term = -self.bias*(self.ring_length-1 == self.current_state 
								and 0 == self.next_state)
		bias_term += -self.bias*(self.ring_length-1 == self.next_state 
								 and 0 == self.current_state)
		return bias_term - np.log(transition_probability / 0.5)

class ring_model_periodic(object):

	def __init__(self, parameters):
		self.ring_length = parameters['ring_length']
		self.bias = parameters['bias']
		self.current_state = 0
		self.next_state = 0
		self.scale = parameters['scale']
		self.constant = parameters['constant']

	def original_transition_probability(self):
		potential = (self.scale*np.sin(2*np.pi*self.current_state/self.ring_length)
					 + self.constant)
		if self.current_state == (self.next_state-1) % self.ring_length:
			probability = np.exp(potential)/(np.exp(potential) + 1)
		else:
			probability = 1/(np.exp(potential) + 1)
		return probability

	def original_transition_query(self, current_state, next_state):
		potential = (self.scale*np.sin(2*np.pi*current_state/self.ring_length)
					 + self.constant)
		if current_state == (next_state-1) % self.ring_length:
			probability = np.exp(potential)/(np.exp(potential) + 1)
		elif current_state == (next_state+1) % self.ring_length:
			probability = 1/(np.exp(potential) + 1)
		else:
			probability = 0
		return probability

	def reward(self, transition_probability):
		if self.current_state == (self.next_state-1) % self.ring_length:
			bias_term = -self.bias
		else:
			bias_term = self.bias
		original_probability = self.original_transition_probability()
		return bias_term - np.log(transition_probability / original_probability)

	def transition_bias(self, current_state, next_state):
		if current_state == (next_state-1) % self.ring_length:
			bias_term = -self.bias
		else:
			bias_term = self.bias
		return np.exp(bias_term)

	def original_dynamics(self):
		probabilities = np.zeros((2, self.ring_length))
		for state in range(self.ring_length):
			potential = (self.scale*np.sin(2*np.pi*state/self.ring_length)
						 + self.constant)
			probabilities[1][state] = np.exp(potential)/(np.exp(potential) + 1)
			probabilities[0][state] = 1/(np.exp(potential) + 1)
		return probabilities