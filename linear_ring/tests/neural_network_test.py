import sys
import os
import time
import numpy as np
import numpy.random
from matplotlib import pyplot
source_path = os.path.join("../source/")
sys.path.insert(0,source_path)
import ring_neural
import ring_model
import ring_exact

init_time = time.time()

model_parameters = dict(
	ring_length = 1000, 
	bias = 0.2, 
	scale = 0.5,
)
model = ring_model.ring_model_periodic(model_parameters)
original_dynamics = model.original_dynamics()

value_parameters = dict(
	first_layer_neurons = 10,
	second_layer_neurons = 5,
)
dynamic_parameters = dict(
	first_layer_neurons = 10,
	second_layer_neurons = 5,
)
dynamics_parameters = dict(
	model = model,
	reward_learning_rate = 0.00003,
	value_learning_rate = 0.00005,
	dynamic_learning_rate = 0.00002,
	total_time = 1000000,
	value_parameters = value_parameters,
	dynamic_parameters = dynamic_parameters,
)
dynamics = ring_neural.ring_dynamics_neural(dynamics_parameters)
average_reward, position = dynamics.run()
final_dynamics = dynamics.dynamics()

print(time.time() - init_time)

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