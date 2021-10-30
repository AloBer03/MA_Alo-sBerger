## Single_Neuron_with_numpy

## code from NNFS
## My own comments are marked with ##
## My own code start with ##-- and ends with --##

import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
		   [0.5, -0.91, 0.26, -0.5],
		   [-0.26, -0.27, 0.17, 0.87]]
bias = [2, 3, 0.5]

layer_outputs = np.dot(weights, inputs) + bias
## Dotmultiplication(matrix, vector) and add bias (vec, vec)
## Numpy intrepret matrix as lists of vector and multiplies it with every vectore from the list inputs
## This works because input is a vector: if batch of input you need to transpose B to be able to do dotmatrix

print(layer_outputs)