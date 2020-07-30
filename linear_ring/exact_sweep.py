import sys
import os
import pickle
import numpy as np
import numpy.random
from scipy import linalg
from matplotlib import pyplot
source_path = os.path.join("source/")
sys.path.insert(0,source_path)
import ring_tabular
import ring_linear
import ring_model
import ring_exact

ring_length = 499
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
bias_step = 0.05
biases = [initial_bias + bias_step*i for i in range(21)]
statistics = ring_exact.biased_statistics(dict(model = model))

scgf = []
for bias in biases:
	print(bias)
	model.bias = bias
	statistics.update_model(model)
	scgf.append(statistics.rate_function)

#np.save('scgf_exact', np.array(scgf))
pyplot.plot(biases, scgf)
pyplot.show()