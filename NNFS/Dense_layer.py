## Dense_layer

## code from NNFS
## My own comments are marked with ##
## My own code start with ##-- and ends with --##

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Class of a fully connected layer (dense Layer)
class Layer_Dense:

	# Initialize (for ex. for preloaded model) here randomly
	def __init__(self, n_inputs, n_neuron):
		# Initialize weights and bias
		self.weights = 0.01 * np.random.randn(n_inputs, n_neuron) # randomly
		# Note inputs then neuron  (not Neuron then input) so we don't need to transpose afterwards
		self.biases = np.zeros((1, n_neuron)) #all zero (can be diffrent)

	#  Forward pass (passing data from one layer to the other)
	def forward(self, inputs):
		# Remember input values
		self.inputs = inputs
		# Calculate output through input, weights, bias
		self.output = np.dot(inputs, self.weights) + self.biases

	# Backward pass
	def backward(self, dvalues):
		# Gradient on parameters
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=0)
		# Gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:

	# Forward pass
	def forward(self, inputs):
		# Remember the inputs values
		self.inputs = inputs
		# Calculate ouput value
		self.output = np.maximum(0, inputs)

	# backward pass
	def backward(self, dvalues):
		# Since we need to modify the originaal variable let's make a copy of the values first
		self.dinputs = dvalues.copy()

		# Zero gradient where input values were nagative
		self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

	# Forward pass
	def forward(self, inputs):

		# Remember inputs value
		self.inputs = inputs

		# Get unnormalized probability	here substracting highest value		  
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True)) 
																			  
		# Normalize them for each sample 
		# Numpy divides each value from each outputbatch by the corresponding row value  # [[sum of batch 1],
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)			#= [sum of batch 2],
																						#  [sum of n-batch]]
		self.output = probabilities

	# Backwards pass
	def backward(self, dvalues):

		# Create uninitialized array
		self.dinputs = np.empty_like(dvalues)

		# Enumerate outputs and gradients
		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
			# Flatten output array
			single_output = single_output.reshape(-1, 1)
			# Calculate Jacobian matrix of the output
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
			# Calculate sample-wise gradient and add it to the array of samples gradients
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


# Common loss class 
class Loss: 

	# Calculates the data and regularization losses
	# Given model output and ground truth values
	def calculate(self, output, y):

		# Calculate sample losses
		sample_losses = self.forward(output, y)

		# Calculate mean loss
		data_loss = np.mean(sample_losses)

		# Return loss
		return data_loss

class Loss_CategoricalCrossentropy(Loss):

	# Frward pass
	def forward(self, y_pred, y_true):
		# Number of samples
		samples =  len(y_pred)

		# Clip data to prevent division by 0
		# Clip both sides to not drag mean towards any value
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		# Probabilities for target values only if categorical labels (one dimension [dog,cat,cat])
		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]

		# Mask values - only for one-hot encoded labels (multiple dimension [[1,0,0], [0,1,0], [0,1,0]])
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

		# Losses
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

	# Backwards pass
	def backward(self, dvalues, y_true):

		# Number of samples
		samples = len(dvalues)
		# Number of labels in ervery sample (use first sample to count them)
		labels = len(dvalues[0])

		# If labels are sparse, turn them into one-hot vector
		if len(y_true.shape) ==1:
			y_true = np.eye(labels)[y_true]

		# Calculate gradient
		self.dinputs = -y_true / dvalues
		# Normalize gradient
		self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backwards step
class Activation_Softmax_Loss_CategoricalCrossentropy():

	# Creates activation and loss function objects
	def __init__(self):
		self.activation = Activation_Softmax() 
		self.loss = Loss_CategoricalCrossentropy()

	# Froward pass
	def forward(self, inputs, y_true):
		# Output layer's activation function
		self.activation.forward(inputs)
		# Set output
		self.output = self.activation.output
		# Calculate and return the loss value
		return self.loss.calculate(self.output, y_true)

	# Backwards pass
	def backward(self, dvalues, y_true):

		# Number of samples
		samples = len(dvalues)
		# If labels are one-hot encoded turn them into discrete values
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)

		# Copy so we can sagely modify 
		self.dinputs = dvalues.copy()
		# Calculate gradient
		self.dinputs[range(samples), y_true] -= 1
		# Normalize gradient
		self.dinputs = self.dinputs / samples


# Creating samples
X, y = spiral_data(samples= 100, classes = 3) # X is input and y is output (if given x it should give y out)

# Create object (dense layer 2 inputs and 3 neurons)
dense1 = Layer_Dense(2, 3)

# Create activatoin ReLU (to use afterwards)
activation1 = Activation_ReLU()

# Creating second dense layer
dense2 = Layer_Dense(3, 3) #3 inputs and 3 Neuron (here then ouputs)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Perform a forward pass (calculating output)
dense1.forward(X)

# Pass ouput of previous layer through activation function
activation1.forward(dense1.output)

# Taking the ouput of the previous layer and pass it through weights and bias
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function 
# Takes the output o second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)

# Let's see output of the first few samples: 
print(loss_activation.output[:5])

# rint loss values
print('loss:', loss)

# Calculate accuracy from outputs of activation2 and targets
# Calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
	y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

# Print accuracy
print('acc:', accuracy)

# Backwards pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print(gradients)
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)



