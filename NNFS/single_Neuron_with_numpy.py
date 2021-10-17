# Single_Neuron_with_numpy

import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
		   [0.5, -0.91, 0.26, -0.5],
		   [-0.26, -0.27, 0.17, 0.87]]
bias = [2, 3, 0.5]

layer_outputs = np.dot(weights, inputs) + bias
# Hier wird punktmultipliziert(matrix, vector) und bias addiert(vec, vec)
# Numpy interpriet matrix als liste von vector und multipliziert mit jedem Vec aus der Liste mit inputs(Vec) 
# This works because input is a vector: if batch of input you need to transpose B to be able to do dotmatrix

print(layer_outputs)