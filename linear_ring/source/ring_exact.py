import sys
import os
import numpy as np
import numpy.random
from scipy import linalg

class biased_statistics(object):

	def __init__(self, parameters):
		self.model = parameters['model']
		self._construct_tilted_operator()
		self._doob_dynamics()

	def _construct_tilted_operator(self):
		self.tilted_master_operator = np.zeros((self.model.ring_length, 
												self.model.ring_length), 
											   dtype = np.complex64)
		for current_state in range(self.model.ring_length):
			for next_state in range(self.model.ring_length):
				self.tilted_master_operator[next_state][current_state] = (
					self.model.original_transition_query(current_state, next_state)
					* self.model.transition_bias(current_state, next_state))

	def _doob_dynamics(self):
		spectrum = linalg.eig(self.tilted_master_operator, left = True)
		sorted_index = spectrum[0].argsort()[::-1]
		eigenvalues = spectrum[0][sorted_index]
		left_eigenvectors = spectrum[1][:,sorted_index]
		right_eigenvectors = spectrum[2][:,sorted_index]

		print(eigenvalues[0])
		print(eigenvalues[-1])
		self.rate_function = np.log(eigenvalues[0])
		norm = np.dot(left_eigenvectors[:,0],right_eigenvectors[:,0])
		left_eigenvectors[:,0] /= norm
		self.stationary_state = (left_eigenvectors[:,0] * right_eigenvectors[:,0])
		self.doob_master_operator = (np.diag(left_eigenvectors[:,0]) 
									 @ self.tilted_master_operator
									 @ np.diag(1/left_eigenvectors[:,0]))/eigenvalues[0]

	def update_model(self, new_model):
		self.model = new_model
		self._construct_tilted_operator()
		self._doob_dynamics()

