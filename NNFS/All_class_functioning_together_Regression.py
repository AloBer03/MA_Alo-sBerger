# All classes functioning together for Regression

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


# Class of a fully connected layer (dense Layer)
class Layer_Dense:

	# Initialize (for ex. for preloaded model) here randomly
	def __init__(self, n_inputs, n_neuron,
				 weight_regularizer_l1=0, weight_regularizer_l2=0,
				 bias_regularizer_l1=0, bias_regularizer_l2=0):
		# Initialize weights and bias
		self.weights = 0.01 * np.random.randn(n_inputs, n_neuron) # randomly
		# Note inputs then neuron  (not Neuron then input) so we don't need to transpose afterwards
		self.biases = np.zeros((1, n_neuron)) #all zero (can be diffrent)
		# Set regularization strength
		self.weight_regularizer_l1 = weight_regularizer_l1
		self.weight_regularizer_l2 = weight_regularizer_l2
		self.bias_regularizer_l1 = bias_regularizer_l1
		self.bias_regularizer_l2 = bias_regularizer_l2

	# Forward pass (passing data from one layer to the other)
	def forward(self, inputs):
		# Remember input values
		self.inputs = inputs
		# Calculate output through input, weights, bias
		self.output = np.dot(inputs, self.weights) + self.biases

	# Backward pass
	def backward(self, dvalues):
		# Gradient on parameters
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
		
		# Gradient on regularization
		# L1 on weights
		if self.weight_regularizer_l1 > 0:
			dL1 = np.ones_like(self.weights)
			dL1[self.weights < 0] = -1
			self.dweights += self.weight_regularizer_l1 * dL1
		# L2 on weights
		if self.weight_regularizer_l2 > 0:
			self.dweights += 2 * self.weight_regularizer_l2 * self.weights

		# L1 on biases
		if self.bias_regularizer_l1 > 0:
			dL1 = np.ones_like(self.biases)
			dL1[self.biases < 0] = -1
			self.dbiases += self.bias_regularizer_l1 * dL1
		# L2 on biases
		if self.bias_regularizer_l2 > 0:
			self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

		# Gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)

# Dropout
class Layer_Dropout:

	# Init
	def __init__(self, rate):
		# Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
		self.rate = 1 - rate

	# Forward pass 
	def forward(self, inputs):

		# Save input values
		self.inputs = inputs
		# Generate and save scaled mask
		self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
		# Apply mask to output values
		self.output = inputs * self.binary_mask

	# Backward pass
	def backward(self, dvalues):
		# Gradient on values
		self.dinputs = dvalues * self.binary_mask


class Activation_ReLU:

	# Forward pass
	def forward(self, inputs):
		# Remember the inputs values
		self.inputs = inputs
		# Calculate ouput value
		self.output = np.maximum(0, inputs)

	# Backward pass
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

# Sigmoid activation
class Activation_Sigmoid:

	# Forward pass
	def forward(self, inputs):
		# Save input and calculate/save output of the sigmoid function
		self.inputs = inputs
		self.output = 1 / (1 + np.exp(-inputs))

	# Backward pass
	def backward(self, dvalues):
		# Derivative - calculates form output of the sigmoid function
		self.dinputs = dvalues * (1 - self.output) * self.output

# Common loss class 
class Loss: 

	# Calculates the data and regularization losses
	# given model output and ground truth values
	def calculate(self, output, y):

		# Calculate sample losses
		sample_losses = self.forward(output, y)

		# Calculate mean loss
		data_loss = np.mean(sample_losses)

		# Return loss
		return data_loss

	# Regularization loss calculation
	def regularization_loss(self, layer):

		# 0 by default
		regularization_loss = 0

		# L1 regularization - weithgts
		# calculate only when factor greater than 0
		if layer.weight_regularizer_l1 > 0:
			regularization_loss += lyer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

		# L2 regularization - weights
		if layer.weight_regularizer_l2 > 0:
			regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

		# L1 regularization - biases
		# calculate only when factor is greater than 0
		if layer.bias_regularizer_l1 > 0:
			regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

		# L2 regularization - biases
		if layer.bias_regularizer_l2 > 0:
			regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

		return regularization_loss

class Loss_CategoricalCrossentropy(Loss):

	# Forward pass
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

# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
	
	# Forward pass
	def forward(self, y_pred, y_true):

		# Clip data to prevent dicision by 0
		# Clip both sides to not drag mean towards any value
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		# Calculate samle-wise loss
		sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
		sample_losses = np.mean(sample_losses, axis= -1)

		# Return losses
		return sample_losses

	# Backward pass
	def backward(self, dvalues, y_true):

		# Number of samples
		samples = len(dvalues)
		# Number of outputs in every sample
		# We'll use the first sample to count them
		outputs = len(dvalues[0])

		# Clip data to prevent division by 0
		# Clip both sides to not drag mean towards any value
		clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

		# Calculate gradient
		self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
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

#SGD optimizer
class Optimizer_SGD:

	# Initialize optimizer - set settings,
	# learning rate of 1. is default for this optimizer
	def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.momentum = momentum

	# Call once before any parameter updates 
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay *self.iterations))

	# Update parameters 
	def update_params(self, layer):
		# Momentum
		# If we use momentum
		if self.momentum:

			# If layer does not contain momentum arrays, create them filled with zeros
			if not hasattr(layer, 'weight_momentums'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				# If there is no momentum array for weights
				# The array doesn't exist for biases yet either
				layer.bias_momentums = np.zeros_like(layer.biases)

			# Build weight updates with momentum - takes prepvious updates multiplied by retain factor
			# and update with current gradient
			weight_updates = self.momentum *layer.weight_momentums - \
							 self.current_learning_rate * layer.dweights
			layer.weight_momentums = weight_updates

			# Build bias update_params
			bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
			layer.bias_momentums = bias_updates

		# Else Vannilla SGD updates (as before momentum update)
		else: 
			weight_updates = -self.current_learning_rate * layer.dweights
			bias_updates = -self.current_learning_rate * layer.dbiases


		# Update weights and biases using either vanilla or momentum updates
		layer.weights += weight_updates
		layer.biases += bias_updates

	# Call once after any parameters 
	def post_update_params(self):
		self.iterations += 1

# AdaGrad optimizer
class Optimizer_Adagrad:

	# Initialize optimizer - set settings,
	# learning rate of 1. is default for this optimizer
	def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		

	#Call once before any parameter updates 
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	# Update parameters 
	def update_params(self, layer):
		# Adagrad: cache

		# If layer does not contain cache arrays, create them filled with zeros
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		# Update cache with squared current gradients
		layer.weight_cache += layer.dweights**2
		layer.bias_cache += layer.dbiases**2
		
		# Vanilla SGD parameter update + normalization with quare rooted cache
		layer.weights += -self.current_learning_rate * \
						 layer.dweights / \
						 (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.current_learning_rate * \
						layer.dbiases / \
						(np.sqrt(layer.bias_cache) + self.epsilon)

	# Call once after any parameters 
	def post_update_params(self):
		self.iterations += 1

# RMSprop optimizer
class Optimizer_RMSprop:

	# Initialize optimizer - set settings,
	# learning rate of 0.001 as default because it already has much momentum
	# rho hyperparameter (decay rate of cache memory)
	def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.rho = rho

	# Call once before any parameter updates 
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay *self.iterations))

	# Update parameters 
	def update_params(self, layer):

		# If layer does not contain cache arrays, create them filled with zeros
		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		# Update cache with squared current gradients
		layer.weight_cache = self.rho * layer.weight_cache + \
							 (1 - self.rho) * layer.dweights**2
		layer.bias_cache = self.rho * layer.bias_cache + \
						   (1 - self.rho) * layer.dbiases**2

		# Vanilla SGD parameter update + normalization
		# with square rooted cache
		layer.weights += -self.current_learning_rate * layer.dweights / \
						 (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.dbiases / \
						(np.sqrt(layer.bias_cache) + self.epsilon)

	# Call once after any parameters 
	def post_update_params(self):
		self.iterations += 1

# Adam
# Adaptive Momentum
# it is RMSProp but with momentum build in
class Optimizer_Adam:

	# Initialize optimizer - set settings
	def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2 = 0.999):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.beta_1 = beta_1
		self.beta_2 = beta_2

	# Call once before any parameter updates 
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay *self.iterations))

	# Update parameters 
	def update_params(self, layer):

		# If layer does not contain momentum and cache arrays, create them filled with zeros
		if not hasattr(layer, 'weight_cache'):
			layer.weight_momentums = np.zeros_like(layer.weights)
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_momentums = np.zeros_like(layer.biases)
			layer.bias_cache = np.zeros_like(layer.biases)

		# Update Momentum with with current gradient
		layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
									(1 - self.beta_1) * layer.dweights
		layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
									(1 - self.beta_1) * layer.dbiases
		# Update the momentum with a fraction of itself plus opposite fraction of the gradient 
		
		# Get corrected momentum
		# self.iteration is 0 at first pass and we need to start with 1 here
		weight_momentums_corrected = layer.weight_momentums / \
										(1 - self.beta_1 ** (self.iterations + 1))
		bias_momentums_corrected = layer.bias_momentums / \
										(1 - self.beta_1 **(self.iterations + 1))
		# At the beginning dividing by smaller number makes the momentum bigger (after time divisor tends to 1)
		# This speeds up the process at the beginning compared to the end (where more finess is required)

		# Update cache with squared current gradients
		layer.weight_cache = self.beta_2 * layer.weight_cache + \
							 (1 - self.beta_2) * layer.dweights ** 2
		layer.bias_cache = self.beta_2 * layer.bias_cache + \
						   (1 - self.beta_2) * layer.dbiases ** 2
		# Get corrected cache here again
		weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
		bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

		# Vanilla SGD parameter update + normalization with square rooted cache
		layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
						 (np.sqrt(weight_cache_corrected) + self.epsilon)
		layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
						(np.sqrt(bias_cache_corrected) + self.epsilon)

	# Call once after any parameters 
	def post_update_params(self):
		self.iterations += 1


# Creating samples 
X, y = spiral_data(samples= 100, classes = 2) # X is input and y is output (if given x it should give y out)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)

# Create object (dense layer 2 input features and 64 output  values)
dense1 = Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)

# Create activatoin ReLU (to use afterwards)
activation1 = Activation_ReLU()

# Creating second dense layer
dense2 = Layer_Dense(64, 1) #64 inputs features and 1 output

# Create Softmax classifier's combined loss and activation
activation2 = Activation_Sigmoid()

# Create loss function
loss_function = Loss_BinaryCrossentropy()

# Create an optimizer
# Optimizer = Optimizer_SGD(decay=8e-8, momentum=0.9)
# Optimizer = Optimizer_Adagrad(decay=1e-4)
optimizer = Optimizer_Adam(decay=5e-7)

# Train in loop
for epoch in range(10001):
	
	# Perform a forward pass (calculating output)
	dense1.forward(X)

	# Pass ouput of previous layer through activation function
	activation1.forward(dense1.output)

	# Perform a forward pass through second Dense Layer
	# taking the ouput of the previous layer and pass it through weights and bias
	dense2.forward(activation1.output)

	# Perform a forward pass through activation function
	# takes the output of second dense layer here
	activation2.forward(dense2.output)

	# Calculate the data loss
	data_loss = loss_function.calculate(activation2.output, y)

	# Calculate regularization penalty
	regularization_loss = loss_function.regularization_loss(dense1) + \
						  loss_function.regularization_loss(dense2)

	# Calculate overall loss
	loss = data_loss + regularization_loss

	# Calculate accuracy from outputs of activation2 and targets
	# Part in the brackets returns a binary mask - array consisting 
	# of True/False values, multiplying it by 1 changes it into array of 1s and 0s
	predictions = (activation2.output > 0.5) * 1
	accuracy = np.mean(predictions == y)

	if not epoch % 100:
		print(f'epoch: {epoch}, ' + 
	  		  f'acc: {accuracy:.3f}, ' +
	  		  f'loss: {loss:.3f} (' +
	  		  f'data_loss: {data_loss:.3f}, ' +
	  		  f'reg_loss: {regularization_loss:.3f}), ' +
	  		  f'lr: {optimizer.current_learning_rate}')

	# Backwards pass
	loss_function.backward(activation2.output, y)
	activation2.backward(loss_function.dinputs)
	dense2.backward(activation2.dinputs)
	activation1.backward(dense2.dinputs)
	dense1.backward(activation1.dinputs)

	# Updare weights and biases
	optimizer.pre_update_params()
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.post_update_params()

# Validate the model

# Create testdataset
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y_test = y_test.reshape(-1, 1)

#perform a forward pass of our testing data through this layer
dense1.forward(X_test)

# Perform a forward pass through second activation function 
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through second Dense layer
# takes outputs of the second dense layer here
activation2.forward(dense2.output)

# Calculate the data loss
loss = loss_function.calculate(activation2.output, y_test)

# Calculate accuracy from output of activation2 and targets
# Part in the brackets returns a binary mask - array consisting of 
# True/False values, multiplyingit by 1 changes it into array of 1s and 2s
predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')