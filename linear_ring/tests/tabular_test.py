import sys
import os
import numpy as np
import numpy.random
from scipy import linalg
from matplotlib import pyplot
source_path = os.path.join("../source/")
sys.path.insert(0,source_path)
import ring_tabular
import ring_model
import ring_exact

model_parameters = dict(
	ring_length = 50, 
	bias = 0, 
	scale = 0.3
)
model = ring_model.ring_model_periodic(model_parameters)
original_dynamics = model.original_dynamics()
dynamics_parameters = dict(
	model = model, 
	reward_learning_rate = 0.0005,
	value_learning_rate = 0.2,
	dynamic_learning_rate = 0.01,
)
dynamics = ring_tabular.ring_dynamics_tabular(dynamics_parameters)
dynamics.initialize(10000)
average_reward, entropy, current, position = dynamics.train(100000)
final_dynamics = dynamics.dynamics()

pyplot.figure(figsize = (40,5))
pyplot.subplot(231)
pyplot.plot(average_reward[::100])
pyplot.subplot(232)
pyplot.plot(entropy[::100])
pyplot.subplot(233)
pyplot.plot(current[::100])
pyplot.subplot(234)
pyplot.plot(position[::100])
pyplot.subplot(235)
pyplot.plot(final_dynamics[:,0], c = 'k', lw = 3)
pyplot.plot(final_dynamics[:,1], c = 'b', lw = 3)
pyplot.plot(original_dynamics[0], ls = '--', c = 'k', lw = 3)
pyplot.plot(original_dynamics[1], ls = '--', c = 'b', lw = 3)
pyplot.ylim(0,1)
pyplot.subplot(236)
pyplot.plot(dynamics.values)

pyplot.show()