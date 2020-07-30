import sys
import os
import numpy as np
import numpy.random
from scipy import linalg
from matplotlib import pyplot
source_path = os.path.join("../source/")
sys.path.insert(0,source_path)
import ring_linear
import ring_model
import ring_exact

model_parameters = dict(
	ring_length = 50, 
	bias = 0.05, 
	scale = 0.1
)
model = ring_model.ring_model_periodic(model_parameters)
original_dynamics = model.original_dynamics()

dynamics_parameters = dict(
	model = model, 
	reward_learning_rate = 0.0003,
	value_learning_rate = 0.02,
	dynamic_learning_rate = 0.005,
	total_time = 50000000,
	value_weight_number = 5,
	dynamic_weight_number = 5,
	discount = 0.99,
)
dynamics = ring_linear.ring_dynamics_fourier(dynamics_parameters)
average_reward, position = dynamics.run()
print(dynamics.potential_approximation.weights)
print(dynamics.value_approximation.weights)
final_dynamics = dynamics.dynamics()

pyplot.figure(figsize = (20,5))
pyplot.subplot(141)
pyplot.plot(average_reward[::100])
pyplot.subplot(142)
pyplot.plot(position[::100])
pyplot.subplot(143)
pyplot.plot(final_dynamics[0], c = 'k', lw = 3)
pyplot.plot(final_dynamics[1], c = 'b', lw = 3)
pyplot.plot(original_dynamics[0], ls = '--', c = 'k', lw = 3)
pyplot.plot(original_dynamics[1], ls = '--', c = 'b', lw = 3)
pyplot.ylim(0,1)
pyplot.subplot(144)
pyplot.plot(dynamics.values())
pyplot.show()