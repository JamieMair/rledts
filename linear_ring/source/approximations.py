import sys
import os
import numpy as np
import numpy.random
from scipy import linalg


class fourier_basis(object):

	def __init__(self, modes, ring_length, initial_weights = 'random'):
		self.modes = modes
		if initial_weights == 'zero':
			self.weights = np.zeros((self.modes*2 + 1))
		elif initial_weights == 'random':
			self.weights = np.random.randn(self.modes*2 + 1)/modes
		self.ring_length = ring_length

	def features(self, state):
		feature_values = np.array([1])
		trig_arguments = ((2 * np.pi * state * np.arange(1, self.modes + 1))
						  / self.ring_length)
		feature_values = np.append(feature_values, np.sin(trig_arguments))
		feature_values = np.append(feature_values, np.cos(trig_arguments))
		return feature_values