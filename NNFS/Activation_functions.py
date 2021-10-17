#Activation Function

import numpy as np

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

# ReLU
output = []
for i in inputs:
	pass
	if i > 0:
		output.append(i)
	else:
		output.append(0)

# Simplyfied with numpy
output = np.maximum(0, inputs)

class Activation_ReLU:

	#forward pass
	def forward(self, inputs):
		#calculate ouput value
		self.output = np.maximum(0, inputs)

layer_outputs = [4.8, 1.21, 2.385]

# Eulers number constant E
E = 2.71828182846 # also possible: math.e

exp_values = []
for output in layer_outputs:
	exp_values.append(E ** output)
print('exponentiatede values: ', exp_values)

# Normalize by dividing by the sum of all numbers 
# All added together = 1 and they range from 0 to 1
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
	norm_values.append(value / norm_base)
print('normalized values:', norm_values)
print('sum of normalized values:', sum(norm_values))

# With numpy
layer_outputs = [4.8, 1.21, 2.385]
# Exponentiate
exp_values = np.exp(layer_outputs)
# Normalize
norm_values = exp_values / np.sum(exp_values)

print("norm values:", norm_values)
print('sum of values:', sum(norm_values))

# Even faster (general)

# Inputs batches
inputs = [[0.1, 0.3, 0.6],
		  [0.3, 0.5, 0.2],
		  [0.02, 0.9, 0.08]]

# Exponential
exp_values = np.exp(inputs)
# Normalize
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Axis 1 refers to y-axis (raws) and 0 to x-axis (columns)
# this means add the values of the specific axis  (if none all numbers are added)
# this is usefull when you have a batch of inputs (our output/input is: [[1,1,1],  1st batch
# so we need to add raws together therefor we use axis=1				 [2,2,2],  2nd batch	
# keepdims is to keep the dimension (so we can divide later on)			  ......])  Nth batch

class Activation_Softmax:

	#forward pass
	def forward(self, inputs):

		#get unnormalized probability										  # [[sum of batch 1],
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True)) #= [sum of batch 2],
																			  #  [sum of n-batch]]
		#normalize them for each sample 
		#numpy divides each value from each outputbatch by the corresponding row value
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

		self.output = probabilities

# Full softmax activation but not much of use because it is significantly faster when combined with loss

# Here added backwards steps
class Activation_Softmax_full:

	#forward pass
	def forward(self, inputs):

		#remember inputs value
		self.inputs = inputs

		#get unnormalized probability	here substracting highest value		  
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True)) 
																			  
		#normalize them for each sample 
		#numpy divides each value from each outputbatch by the corresponding row value  # [[sum of batch 1],
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)			#= [sum of batch 2],
																						#  [sum of n-batch]]
		self.output = probabilities

	#backwards pass
	def backward(self, dvalues):

		#create uninitialized array
		self.dinputs = np.empty_like(dvalues)

		#enumerate outputs and gradients
		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
			#flatten output array
			single_output = single_output.reshape(-1, 1)
			# calculate Jacobian matrix of the output
			jacobian_matrix = np.diagflat(signle_output) - np.dot(single_output, single_output.T)
			#calculate sample-wise gradient and add it to the array of samples gradients
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
