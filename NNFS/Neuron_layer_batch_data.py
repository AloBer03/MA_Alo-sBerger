## Neuron layer + batch of Data

## code from NNFS
## My own comments are marked with ##
## My own code start with ##-- and ends with --##

import numpy as np
import nnfs
from nnfs.datasets import spiral_data ## Gives a batch of data (for convenience)
# A slight variance of the function from https://cs231n.github.io/neural-networks-case-study/

nnfs.init()
## Sets random seed to 0 by default \
## Float type 32 by default	       |-> to ensure we get the same result everytime (easier to check)
## Overwrite original np dotmatrix  /

inputs = [[1.0,2.0,3.0,2.5], 	 		## x = number of inputs
		  [2.0, 5.0, -1.0, 2.0], 		## y = batch of inputs
		  [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],     	## x = number of Neuron/Input on othe previous layer
		   [0.5, -0.91, 0.26, -0.5],  	## y= number of Neuron on the next layer
		   [-0.26, -0.27, 0.17, 0.87]]
bias = [2.0, 3.0, 0.5]
weights2 = [[0.1, -0.14, 0.5],     		## x = number of Neuron/Input on othe previous layer
		   [-0.5, 0.12, -0.33],  		## y= number of Neuron on the next layer
		   [-0.44, 0.73, -0.13]]
bias2 = [-1.0, 2.0, -0.5]

layer1_outputs =  np.dot(inputs, np.array(weights).T) + bias
## Np converts list into matrices and array (bias) and does the operation accordingly 

# Give a list with layer of output (output raw) for each sample of Data (input raw)
# Output and not Neuron because this way we can continue for next layers
layer2_outputs =  np.dot(layer1_outputs, np.array(weights2).T) + bias2

print(layer2_outputs)

# Class of a fully connected layer (dense Layer)
class Layer_denseL:

	# Initialize (for ex. for preloaded model) here randomly
	def __init__(self, n_inputs, n_neuron):

		# Initialize weights and bias
		self.weights = 0.01 * np.random.randn(n_inputs, n_neuron) # randomly
		# Note inputs then neuron  (not Neuron then input) so we don't need to transpose afterwards
		self.biases = np.zeros((1, n_neuron)) #all zero (can be diffrent)
		pass

	# Forward pass (passing data from one layer to the other)
	def forward(self, inputs):
		
		# Calculate output through input, weights, bias
		self.ouput = np.dot(inputs, self.weights) + self.biases
		pass


