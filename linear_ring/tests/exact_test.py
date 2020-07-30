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
	ring_length = 1000, 
	bias = 0.05, 
	scale = 0.1
)
model = ring_model.ring_model_periodic(model_parameters)
biased_statistics = ring_exact.biased_statistics(dict(model = model))
print(biased_statistics.rate_function)
pyplot.figure(figsize = (10,5))
pyplot.subplot(121)
pyplot.plot(np.diag(biased_statistics.doob_master_operator, -1), lw = 3)
pyplot.plot(np.diag(biased_statistics.doob_master_operator, 1), lw = 3)
pyplot.ylim(0,1)
pyplot.subplot(122)
pyplot.plot(biased_statistics.stationary_state)
#pyplot.yscale('log')
pyplot.show()
"-0.0249999470236"