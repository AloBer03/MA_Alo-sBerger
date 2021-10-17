# Single Neuron

import test


inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
		   [0.5, -0.91, 0.26, -0.5],
		   [-0.26, -0.27, 0.17, 0.87]]
bias = [2, 3, 2.5]

# Previous Output
layer_outputs = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, bias):
	# Initialising the output to back to zero everytime
	neuron_output = 0

	# For each prevouis Neuron add weights
	for n_inputs, weight in zip(inputs, neuron_weights):
		neuron_output += n_inputs*weight
	neuron_output = test.dotproduct(inputs, neuron_weights)

	# Add bias at the end
	neuron_output += neuron_bias
	# Add the output to the output list
	layer_outputs.append(neuron_output)



print(layer_outputs)

