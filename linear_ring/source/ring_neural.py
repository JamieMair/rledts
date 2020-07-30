import sys
import os
import math
import numpy as np
import numpy.random
import torch
import torch.nn as neural_net
import torch.nn.functional as functions
import torch.optim as optimizers

class dynamic_approximation(neural_net.Module):

	def __init__(self, parameters):
		super(dynamic_approximation, self).__init__()
		self.hidden1 = neural_net.Linear(1, parameters['first_layer_neurons'])
		#self.hidden2 = neural_net.Linear(parameters['first_layer_neurons'], 
		#						 parameters['second_layer_neurons'])
		self.hidden3 = neural_net.Linear(parameters['first_layer_neurons'], 1)

	def forward(self, state):
		tensor_state = torch.tensor([state], dtype = torch.float)
		intermediate = functions.elu(self.hidden1(tensor_state))
		#intermediate = functions.elu(self.hidden2(intermediate))
		right_probability = torch.sigmoid(self.hidden3(intermediate))
		return right_probability

class value_approximation(neural_net.Module):

	def __init__(self, parameters):
		super(value_approximation, self).__init__()
		self.hidden1 = neural_net.Linear(1, parameters['first_layer_neurons'])
		#self.hidden2 = neural_net.Linear(parameters['first_layer_neurons'], 
		#						 parameters['second_layer_neurons'])
		self.hidden3 = neural_net.Linear(parameters['first_layer_neurons'], 1)

	def forward(self, state):
		tensor_state = torch.tensor([state], dtype = torch.float)
		intermediate = functions.elu(self.hidden1(tensor_state))
		#intermediate = functions.elu(self.hidden2(intermediate))
		value = self.hidden3(intermediate)
		return value

class ring_dynamics_neural(object):

	def __init__(self, parameters):
		self.model = parameters['model']
		self.average_reward = 0
		self.reward_learning_rate = parameters['reward_learning_rate']
		self.value_approximation = value_approximation(parameters['value_parameters'])
		self.value_learning_rate = parameters['value_learning_rate']
		self.value_optimizer = optimizers.SGD(self.value_approximation.parameters(),
											   lr = self.value_learning_rate,
											   weight_decay = 0.01)
		self.dynamic_approximation = dynamic_approximation(
			parameters['dynamic_parameters'])
		self.dynamic_learning_rate = parameters['dynamic_learning_rate']
		self.dynamic_optimizer = optimizers.SGD(self.dynamic_approximation.parameters(),
												 lr = self.dynamic_learning_rate,
												 weight_decay = 0.01)
		self.total_time = parameters['total_time']
		self.time = 0
		self.right_probability = 0

	def run(self):
		average_reward_vs_time = np.zeros((self.total_time + 1))
		position_vs_time = np.zeros((self.total_time + 1))
		position_vs_time[0] = self.model.current_state
		while self.time < self.total_time:
			transition_probability = self._transition()
			self._update(transition_probability)
			self.model.current_state = self.model.next_state
			self.time += 1
			average_reward_vs_time[self.time] = self.average_reward
			position_vs_time[self.time] = self.model.current_state
		return average_reward_vs_time, position_vs_time

	def _transition(self):
		self.right_probability = self.dynamic_approximation.forward(
			self.model.current_state / self.model.ring_length)
		random = numpy.random.random()
		if random < self.right_probability:
			self.model.next_state = (self.model.current_state+1) % self.model.ring_length
			return self.right_probability
		else:
			self.model.next_state = (self.model.current_state-1) % self.model.ring_length
			return 1 - self.right_probability

	def _update(self, transition_probability):
		reward = self.model.reward(transition_probability.item())
		current_value = self.value_approximation.forward(self.model.current_state 
														 / self.model.ring_length)
		next_value = self.value_approximation.forward(self.model.next_state 
													  / self.model.ring_length)
		td_error = (next_value + reward - self.average_reward - current_value).item()
		self.average_reward += self.reward_learning_rate * td_error
		self.value_optimizer.zero_grad()
		(-td_error * current_value).backward()
		self.value_optimizer.step()
		self.dynamic_optimizer.zero_grad()
		error_term = (-td_error * math.exp(-0.1*abs(td_error)) 
					  * torch.log(transition_probability))
		#error_term = -td_error * torch.log(transition_probability)
		error_term.backward()
		self.dynamic_optimizer.step()

	def dynamics(self):
		probabilities = np.zeros((2, self.model.ring_length))
		with torch.no_grad():
			for state in range(self.model.ring_length):
				right_probability = self.dynamic_approximation.forward(
					state / self.model.ring_length)
				probabilities[1][state] = right_probability
				probabilities[0][state] = 1 - right_probability
		return probabilities

	def values(self):
		values = np.zeros((self.model.ring_length))
		with torch.no_grad():
			for state in range(self.model.ring_length):
				values[state] = self.value_approximation.forward(
					state / self.model.ring_length)
		return values