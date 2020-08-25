import sys
import os
import pickle
import time
import numpy as np
import numpy.random
from scipy import linalg
from matplotlib import pyplot
source_path = os.path.join("source/")
sys.path.insert(0,source_path)
import ring_linear
import ring_model

ring_length = 1000
initial_bias = -0.25
scale = 0.3
constant = 0.15

model_parameters = dict(
	ring_length = ring_length, 
	bias = initial_bias, 
	scale = scale,
	constant = constant,
)
model = ring_model.ring_model_periodic(model_parameters)
original_dynamics = model.original_dynamics()
dynamics_parameters = dict(
	model = model, 
	reward_learning_rate = 0.0005,
	entropy_learning_rate = 0.00005,
	current_learning_rate = 0.00005,
	value_learning_rate = 0.03,
	dynamic_learning_rate = 0.01,
	weight_number = 4,
	discount = 0.999,
)
discounted = False
if discounted:
	dynamics = ring_linear.ring_dynamics_fourier_discounted(dynamics_parameters)
else:
	dynamics = ring_linear.ring_dynamics_fourier(dynamics_parameters)
#dynamics.initialize(10000)

bias_step = 0.01
bias_steps = 101
biases = [initial_bias + bias_step*i for i in range(bias_steps)]
scgf = []
entropy = []
current = []
potentials = []
values = []
stat_states = []

train_steps = 1000000
eval_steps = 1000000

past_time = time.time()
for bias in biases:
	dynamics.model.bias = bias
	dynamics.train(train_steps)
	dynamics.evaluate(eval_steps)
	scgf.append(dynamics.average_reward)
	entropy.append(dynamics.entropy)
	current.append(dynamics.current)
	potentials.append(dynamics.potential())
	vals = dynamics.values()
	stat_state = dynamics.stationary_state()
	values.append(vals  - np.dot(vals, stat_state))
	stat_states.append(stat_state)
print(time.time() - past_time)

save = True
run = 3
if save:
	if not os.path.isdir("data/run_%s"%(run)):
		os.mkdir("data/run_%s"%(run))
	np.save("data/run_%s/scgf"%(run), scgf)
	np.save("data/run_%s/entropy"%(run), entropy)
	np.save("data/run_%s/current"%(run), current)
	np.save("data/run_%s/stationary_state"%(run), 
			np.array(stat_states, dtype = np.float64))
	np.save("data/run_%s/potential"%(run), np.array(potentials))
	np.save("data/run_%s/values"%(run), np.array(values, dtype = np.float64))
	f = open("data/run_%s/params.txt"%(run), "w+")
	f.write("Ring length: %s\n"%(model_parameters['ring_length']))
	f.write("Initial bias: %s\n"%(initial_bias))
	f.write("Bias step: %s\n"%(bias_step))
	f.write("Bias steps: %s\n"%(bias_steps))
	f.write("Scale: %s\n"%(model_parameters['scale']))
	f.write("Constant: %s\n"%(model_parameters['constant']))
	f.write("Reward learning rate: %s\n"%(dynamics_parameters['reward_learning_rate']))
	f.write("Entropy learning rate: %s\n"%(dynamics_parameters['entropy_learning_rate']))
	f.write("Current learning rate: %s\n"%(dynamics_parameters['current_learning_rate']))
	f.write("Value learning rate: %s\n"%(dynamics_parameters['value_learning_rate']))
	f.write("Dynamic learning rate: %s\n"%(dynamics_parameters['dynamic_learning_rate']))
	f.write("Weight number: %s\n"%(dynamics_parameters['weight_number']))
	if discounted:
		f.write("Discount: %s\n"%(dynamics_parameters['discount']))
	f.write("Training steps: %s\n"%(train_steps))
	f.write("Evaluation steps: %s\n"%(eval_steps))
	f.close()
	

pyplot.figure(figsize = (10,4))
pyplot.subplot(231)
pyplot.plot(biases, scgf)
pyplot.subplot(232)
pyplot.plot(biases, entropy)
pyplot.subplot(233)
pyplot.plot(biases, current)
pyplot.subplot(234)
#pyplot.plot(np.array(potentials).T, c = 'k', lw = 3)
#pyplot.plot(np.log(original_dynamics[1]/original_dynamics[0]), 
#			ls = '--', c = 'b', lw = 3)
pyplot.pcolor(np.array(stat_states, dtype = np.float64))
pyplot.subplot(235)
pyplot.pcolor(np.array(potentials))
pyplot.subplot(236)
pyplot.pcolor(np.array(values, dtype = np.float64))

pyplot.show()