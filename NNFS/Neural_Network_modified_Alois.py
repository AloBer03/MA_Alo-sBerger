## Neural_Network_stats
# Main

## code from NNFS
## My own comments are marked with ##
## My own code start with ##-- and ends with --##

## Adding a functino to Model, which output stats about the model
## To remember the stats we store the stats when initiating the classes
## (For Loss and Accuracy it stores the stats during the forward pass)
## During Training we store the learningrate, accuracy and the loss 

## First just print what our Model structur is and what it parametrs are
## Then we display the Learingrate, Accuracy and Loss graphs
## At the end we also print all the weights and biases (from red (highest value) to blue (lowest value))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import copy
import pickle
import numpy as np
import nnfs
import os
import cv2


nnfs.init()

# Dense Layer
class Layer_Dense:

	# Layer initialization
	def __init__(self, n_inputs, n_neuron,
				 weight_regularizer_l1=0, weight_regularizer_l2=0,
				 bias_regularizer_l1=0, bias_regularizer_l2=0):
		# Initialize weights and bias
		self.weights = 0.01 * np.random.randn(n_inputs, n_neuron)
		self.biases = np.zeros((1, n_neuron)) 
		# Set regularization strength
		self.weight_regularizer_l1 = weight_regularizer_l1
		self.weight_regularizer_l2 = weight_regularizer_l2
		self.bias_regularizer_l1 = bias_regularizer_l1
		self.bias_regularizer_l2 = bias_regularizer_l2
		##-- Store stats
		self.stat = 'Layer_Dense: '+str(n_inputs)+', '+str(n_neuron)
		# --#

	# Forward pass
	def forward(self, inputs, training):
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

	# Retrieve layer parameters
	def get_parameters(self):
		return self.weights, self.biases

	# Set weights and biases in a layer instance
	def set_parameters(self, weights, biases):
		self.weights = weights
		self.biases = biases

# Dropout
class Layer_Dropout:

	# Init
	def __init__(self, rate):
		# Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
		self.rate = 1 - rate
		##-- Store stats
		self.stat = "Layer_Dropout: rate:"+str(rate)
		# --##

	# Forward pass 
	def forward(self, inputs, training):

		# Save input values
		self.inputs = inputs

		# If not in the training mode - return values
		if not training:
			self.output = inputs.copy()
			return

		# Generate and save scaled mask
		self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
		# Apply mask to output values
		self.output = inputs * self.binary_mask

	# Backward pass
	def backward(self, dvalues):
		# Gradient on values
		self.dinputs = dvalues * self.binary_mask

# Input "layer"
class Layer_Input:

	# Forward pass
	def forward(self, inputs, training):
		self.output = inputs

class Activation_ReLU:

	# Forward pass
	def forward(self, inputs, training):
		# Remember the inputs values
		self.inputs = inputs
		# Calculate ouput value from inputs
		self.output = np.maximum(0, inputs)
		##-- Store stats
		self.stat = "Activation_ReLU"
		# --##

	# Backward pass
	def backward(self, dvalues):
		# Since we need to modify the originaal variable, let's make a copy of the values first
		self.dinputs = dvalues.copy()

		# Zero gradient where input values were nagative
		self.dinputs[self.inputs <= 0] = 0

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return outputs

# Softmax activation
class Activation_Softmax:

	# Forward pass
	def forward(self, inputs, training):

		# Remember input values
		self.inputs = inputs

		# Get unnormalized probabilities		  
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
																			  
		# Normalize them for each sample 
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

		self.output = probabilities

		##-- Store stats
		self.stat = 'Activation_Softmax'
		# --##

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
			# Calculate sample-wise gradient and add it to the array of sample gradients
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return np.argmax(outputs, axis=1)

	# Return the confidences (Aloïs)
	def confidencces(self, outputs):
		return outputs

# Sigmoid activation
class Activation_Sigmoid:

	# Forward pass
	def forward(self, inputs, training):
		# Save input and calculate/save output of the sigmoid function
		self.inputs = inputs
		self.output = 1 / (1 + np.exp(-inputs))
		##-- Store stats
		self.stat = 'Activation_Sigmoid'
		# --##

	# Backward pass
	def backward(self, dvalues):
		# Derivative - calculates form output of the sigmoid function
		self.dinputs = dvalues * (1 - self.output) * self.output

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return (outputs > 0.5) * 1

# Linear activation
class Activation_Linear:

	# Forward pass
	def forward(self, inputs, training):
		# Just remember values
		self.inputs = inputs
		self.output = inputs
		##-- Store stats
		self.stat = 'Activation_Linear'
		# --##

	# Backward pass
	def backward(self, dvalues):
		# The derivative is 1, 1 * dvalues = dvalues - the chain rule
		self.dinputs = dvalues.copy()

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return outputs

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
		##-- Store stats
		self.stat = 'Optimizer_SGD'
		# --##

	# Call once before any parameter updates 
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	# Update parameters 
	def update_params(self, layer):

		# If we use momentum
		if self.momentum:
			# If layer does not contain momentum arrays, create them filled with zeros
			if not hasattr(layer, 'weight_momentums'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				# If there is no momentum array for weights
				# The array doesn't exist for biases yet either
				layer.bias_momentums = np.zeros_like(layer.biases)

			# Build weight updates with momentum - takes previous updates multiplied by retain factor
			# and update with current gradients
			weight_updates = self.momentum *layer.weight_momentums - \
							 self.current_learning_rate * layer.dweights
			layer.weight_momentums = weight_updates

			# Build bias updates
			bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
			layer.bias_momentums = bias_updates

		# Vannilla SGD updates (as before momentum update)
		else: 
			weight_updates = -self.current_learning_rate * layer.dweights
			bias_updates = -self.current_learning_rate * layer.dbiases

		# Update weights and biases using either vanilla or momentum updates
		layer.weights += weight_updates
		layer.biases += bias_updates

	# Call once after any parameter updates 
	def post_update_params(self):
		self.iterations += 1

# Adagrad optimizer
class Optimizer_Adagrad:

	# Initialize optimizer - set settings
	def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		##-- Store stats
		self.stat = 'Optimizer_Adagrad'
		# --##
		

	# Call once before any parameter updates 
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	# Update parameters 
	def update_params(self, layer):

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

	# Call once after any parameter updates
	def post_update_params(self):
		self.iterations += 1

# RMSprop optimizer
class Optimizer_RMSprop:

	# Initialize optimizer - set settings
	def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.rho = rho
		##-- Store stats
		self.stat = 'Optimizer_RMSprop'
		# --##

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

	# Call once after any parameter updates
	def post_update_params(self):
		self.iterations += 1

# Adam optimizer
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
		##-- Store stats
		self.stat = 'Optimizer_Adam'
		# --##

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

		# Update Momentum with with current gradients
		layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
									(1 - self.beta_1) * layer.dweights
		layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
									(1 - self.beta_1) * layer.dbiases		
		# Get corrected momentum
		# self.iteration is 0 at first pass and we need to start with 1 here
		weight_momentums_corrected = layer.weight_momentums / \
										(1 - self.beta_1 ** (self.iterations + 1))
		bias_momentums_corrected = layer.bias_momentums / \
										(1 - self.beta_1 **(self.iterations + 1))
		# Update cache with squared current gradients
		layer.weight_cache = self.beta_2 * layer.weight_cache + \
							 (1 - self.beta_2) * layer.dweights ** 2
		layer.bias_cache = self.beta_2 * layer.bias_cache + \
						   (1 - self.beta_2) * layer.dbiases ** 2
		# Get corrected cache
		weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
		bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

		# Vanilla SGD parameter update + normalization with square rooted cache
		layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
						 (np.sqrt(weight_cache_corrected) + self.epsilon)
		layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
						(np.sqrt(bias_cache_corrected) + self.epsilon)

	# Call once after any parameter updates 
	def post_update_params(self):
		self.iterations += 1

# Common loss class 
class Loss:

	# Regularization loss calculation
	def regularization_loss(self):

		# 0 by default
		regularization_loss = 0

		# Calculate regularization loss
		# iterate all trainable layers
		for layer in self.trainable_layers:

			# L1 regularization - weithgts
			# calculate only when factor greater than 0
			if layer.weight_regularizer_l1 > 0:
				regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

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

	# Set/remember trainable layers
	def remember_trainable_layers(self, trainable_layers):
		self.trainable_layers = trainable_layers

	# Calculates the data and regularization losses
	# given model output and ground truth values
	def calculate(self, output, y, *, include_regularization=False):

		# Calculate sample losses
		sample_losses = self.forward(output, y)

		# Calculate mean loss
		data_loss = np.mean(sample_losses)

		# Add accumulated sum of losses and sample count
		self.accumulated_sum += np.sum(sample_losses)
		self.accumulated_count += len(sample_losses)

		# If just data loss - return it
		if not include_regularization:
			return data_loss

		# Return the data and regularization losses
		return data_loss, self.regularization_loss()

	# Calculate accumulated loss
	def calculate_accumulated(self, *, include_regularization=False):

		# Calculate mean loss
		data_loss = self.accumulated_sum / self.accumulated_count

		# If just data loss - return it
		if not include_regularization:
			return data_loss

		# return the data and regularization losses
		return data_loss, self.regularization_loss()

	# Reset variables for accumulated loss
	def new_pass(self):
		self.accumulated_sum = 0
		self.accumulated_count = 0

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

	# Forward pass
	def forward(self, y_pred, y_true):

		##-- Store stats
		self.stat = 'Loss_CategoricalCrossentropy'
		# --##

		# Number of samples in a batch
		samples =  len(y_pred)

		# Clip data to prevent division by 0
		# Clip both sides to not drag mean towards any value
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		# Probabilities for target values - only if categorical labels 
		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]

		# Mask values - only for one-hot encoded labels 
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

		# Losses
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

	# Backwards pass
	def backward(self, dvalues, y_true):

		# Number of samples
		samples = len(dvalues)
		# Number of labels in ervery sample
		# We'll use the first sample to count them
		labels = len(dvalues[0])

		# If labels are sparse, turn them into one-hot vector
		if len(y_true.shape) == 1:
			y_true = np.eye(labels)[y_true]

		# Calculate gradient
		self.dinputs = -y_true / dvalues
		# Normalize gradient
		self.dinputs = self.dinputs / samples

# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

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

# Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
	
	# Forward pass
	def forward(self, y_pred, y_true):

		##-- Store stats
		self.stat = 'Loss_BinaryCrossentropy'
		# --##

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

# Mean Squarred Error loss
class Loss_MeanSquaredError(Loss):
	
	# Forward pass
	def forward(self, y_pred, y_true):

		##-- Store stats
		self.stat = 'Loss_MeanSquaredError'
		# --##

		# Calculate loss
		sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

		# Return losses
		return sample_losses

	# Backward pass
	def backward(self, dvalues, y_true):

		# Number of samples
		samples = len(dvalues)
		# Number of outputs in every sample
		# We'll use the first sample to count them
		outputs = len(dvalues[0])

		# Gradient on values
		self.dinputs = -2 * (y_true - dvalues) / outputs
		# Normalize gradient
		self.dinputs = self.dinputs / samples

# Mean Absolute Error loss
class Loss_MeanAbsoluteError(Loss):
	
	# Forward pass
	def forward(self, y_pred, y_true):

		##-- Store stats
		self.stat = 'Loss_MeanAbsoluteError'
		# --##

		# Calculate loss
		sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

		# Return losses
		return sample_losses

	# Backward pass
	def backward(self, dvalues, y_true):

		# Number of samples
		samples = len(dvalues)
		# Number of outputs in every sample
		# We'll use the first sample to count them
		outputs = len(dvalues[0])

		# Calculate gradient
		self.dinputs = np.sign(y_true - dvalues) / outputs
		# Normalize gradient 
		self.dinputs = self.dinputs / samples

# Common accuracy class
class Accuracy:

	# Calculate an accuracy
	# given predictions and ground truth values
	def calculate(self, predictions, y):

		# Get comparison results
		comparisons = self.compare(predictions, y)

		# Calculate an accuracy
		accuracy = np.mean(comparisons)

		# Add accumulated sum of matching values and sample count
		self.accumulated_sum += np.sum(comparisons)
		self.accumulated_count += len(comparisons)

		# Return accuracy
		return accuracy

	# Calculate accumulated accuracy
	def calculate_accumulated(self):

		# Calculate an accuracy
		accuracy = self.accumulated_sum / self.accumulated_count

		# Return the data and regularization losses
		return accuracy

	# Reset variables for accumulated accuracy
	def new_pass(self):
		self.accumulated_sum = 0
		self.accumulated_count = 0

# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):

	def __init__(self, *, binary=False):
		# Binary model?
		self.binary = binary
		##-- Store stats
		self.stat = 'Accuracy_Categorical'
		# --##

	# No initialization is needed
	def init(self, y):
		# Needs to exist because it's called automatically
		pass

	# Compares predictions to the ground truth values
	def compare(self, predictions, y):
		if not self.binary and len(y.shape) == 2:
			y = np.argmax(y, axis=1)
		return predictions == y

# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):

	def __init__(self):
		# Create precision property
		self.precision = None
		##-- Store stats
		self.stat = 'Accuracy_Regression'
		# --##

	# Calculates precision value based on passed-in ground truth values
	def init(self, y, reinit=False):
		if self.precision is None or reinit:
			self.precision = np.std(y) / 250

	# Compares predictions to the ground truth values
	def compare(self, predictions, y):
		return np.absolute(predictions - y) < self.precision

# Model class
class Model:

	def __init__(self):
		# Create a list of network objects
		self.layers = []
		# Softmax calssifier's output object
		self.softmax_classifier_output = None

	# Add objects to the model
	def add(self, layer):
		self.layers.append(layer)

	# Set loss, optimizer and accuracy
	def set(self, *, loss=None, optimizer=None, accuracy=None): 
		
		if loss is not None:
			self.loss = loss

		if optimizer is not None:
			self.optimizer = optimizer
		
		if accuracy is not None:
			self.accuracy = accuracy

	# Finalize the model
	def  finalize(self):

		# Create and set the input layer
		self.input_layer = Layer_Input()

		# Count all the objects
		layer_count = len(self.layers)

		# Initialize a list containing trainable layers:
		self.trainable_layers = []

		# Iterate the objects
		for i in range(layer_count):

			# If it's the first layer
			# the previous layer object is the input layer
			if i==0:
				self.layers[i].prev = self.input_layer
				self.layers[i].next = self.layers[i+1]

			# All layers except for the first and the last
			elif i < layer_count -1:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.layers[i+1]

			# The last layer - the next object is the loss
			# Also let's save aside the reference to the last object whose output is the model's output
			else:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.loss
				self.output_layer_activation = self.layers[i]

			# If layer contains an attribute called "weights", it's a trainable alyer - 
			# add it to the list of trainable layers
			# We don't need to check for biases - checking for weights is enough
			if hasattr(self.layers[i], 'weights'):
				self.trainable_layers.append(self.layers[i])

		# Update loss object with trainable layers
		if self.loss is not None:
			self.loss.remember_trainable_layers(self.trainable_layers)

		# If output activation is Softmax and loss function is Categorical Cross-Entropy
		# create an object of combined activation and loss function containing
		# faster gradient calculation
		if isinstance(self.layers[-1], Activation_Softmax) and \
		   isinstance(self.loss, Loss_CategoricalCrossentropy):
			# Create an object of combined activation and loss functions
			self.softmax_classifier_output = \
				Activation_Softmax_Loss_CategoricalCrossentropy()

	# Train the model
	def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

		# Initialize accuracy object
		self.accuracy.init(y)

		# Default value if batch size is not set
		train_steps = 1

		# If there is validation data passed, set default number of steps for validation as well
		if validation_data is not None:
			validation_steps = 1

			# For better readability
			X_val, y_val = validation_data

		# Calculate number of steps
		if batch_size is not None:
			train_steps = len(X) // batch_size
			# Dividing rounds down. If there are some remaining data, but not a full batch,
			# this won't include it. Add 1 to include this not full batch
			if train_steps * batch_size < len(X):
				train_steps += 1

			if validation_data is not None:
				validation_steps = len(X_val) // batch_size
				#  Dividing rounds down. If there are some remaining data, but not a full batch,
				# this won't include it. Add 1 to include this not full batch
				if validation_steps * batch_size < len(X_val):
					validation_steps += 1

		# Main training loop
		for epoch in range(1, epochs+1):

			# Prit epoch number
			print(f'epoch: {epoch}')

			# Reset accumulated values in loss and accuracy objects
			self.loss.new_pass()
			self.accuracy.new_pass()

			# Iterate over steps
			for step in range(train_steps):

				# If batch size is not set - train using one step and full dataset
				if batch_size is None:
					batch_X = X
					batch_y = y 

				# Otherwise slice a batch
				else:
						batch_X = X[step*batch_size:(step+1)*batch_size]
						batch_y = y[step*batch_size:(step+1)*batch_size]

				# Perform the forward pass
				output = self.forward(batch_X, training=True)

				# Calculate loss
				data_loss, regularization_loss = \
					self.loss.calculate(output, batch_y, include_regularization=True)
				loss = data_loss + regularization_loss

				# Get predictions and calculate an accuracy
				predictions = self.output_layer_activation.predictions(output)
				accuracy = self.accuracy.calculate(predictions, batch_y)
				
				# Perform a backward pass
				self.backward(output, batch_y)

				# Optimize (update parameters)
				self.optimizer.pre_update_params()
				for layer in self.trainable_layers:
					self.optimizer.update_params(layer)
				self.optimizer.post_update_params()

				# Print a summary
				if not step % print_every or step ==  train_steps - 1:
					print(f'step: {step}, ' +
						  f'acc: {accuracy:.3f}, ' +
						  f'loss: {loss:.3f} (' +
						  f'data_loss: {data_loss:.3f}, ' +
						  f'reg_loss: {regularization_loss:.3f}), ' +
						  f'lr: {self.optimizer.current_learning_rate}')

				##-- Store stats for overall summary
				loss_list.append(loss)
				accuracy_list.append(accuracy)
				lr_list.append(self.optimizer.current_learning_rate)
				# --##

			# Get and print epoch loss and accuracy
			epoch_data_loss, epoch_regularization_loss = \
				self.loss.calculate_accumulated(include_regularization=True)
			epoch_loss = epoch_data_loss + epoch_regularization_loss
			epoch_accuracy = self.accuracy.calculate_accumulated()

			print(f'training, ' +
				  f'acc: {epoch_accuracy:.3f}, ' +
				  f'loss: {epoch_loss:.3f} (' +
				  f'data_loss: {epoch_data_loss:.3f}, ' +
				  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
				  f'lr: {self.optimizer.current_learning_rate}')


			# If there is the validation data
			if validation_data is not None:

				# Evaluate the model
				self.evaluate(*validation_data, batch_size=batch_size)

	# Performs forward pass
	def forward(self, X, training):

		# Call forward method on the input layer this will set the output property that
		# the first layer in "prev" object is expecting
		self.input_layer.forward(X, training)

		# Call forward method of every object in a chain 
		# Pass output of the previous object as a parameter
		for layer in self.layers:
			layer.forward(layer.prev.output, training)

		# "layer" is now the last object from the list
		# return its output
		return layer.output

	# Performs backward pass
	def backward(self, output, y):

		# If softmax classifier
		if self.softmax_classifier_output is not None:
			# First call backward method on the combined activation/loss
			# this will set dinputs properly
			self.softmax_classifier_output.backward(output, y)

			# Since we'll not call backward method of the last layer
			# which is Softmax activation as we used combined activation/loss
			# object, let's set dinputs in this object 
			self.layers[-1].dinputs = \
				self.softmax_classifier_output.dinputs

			# Call backward method going through all the objects but last
			# in reversed order passing dinputs as a parameter
			for layer in reversed(self.layers[:-1]):
				layer.backward(layer.next.dinputs)

			return

		# First call backward method on the loss this will set dinputs property
		# that the last layer will try to access shortly
		self.loss.backward(output, y)

		# Call backward method going through all the objects in reversed order
		# passing dipunpts as a parameter
		for layer in reversed(self.layers):
			layer.backward(layer.next.dinputs)

	# Evaluates the model using passed-in dataset
	def evaluate(self, X_val, y_val, *, batch_size=None):

		# Default value if batch size is not being set
		validation_steps = 1

		# Calculate number of steps
		if batch_size is not None:
			validation_steps = len(X_val) // batch_size
			# Dividing rounds down. If there are some remaining data,
			# but not a full batch, this won't include it 
			# Add '1' to include this not full batch
			if validation_steps * batch_size < len(X_val):
				validation_steps += 1

		# Reset accumulated values in loss and accuracy objects
		self.loss.new_pass()
		self.accuracy.new_pass()

		# Iterate over steps
		for step in range(validation_steps):

			# If batch size is not set - train using one step and full dataset
			if batch_size is None:
				batch_X = X_val
				batch_y = y_val

			# Otherwise slice a batch
			else:
				batch_X = X_val[step*batch_size:(step+1)*batch_size]
				batch_y = y_val[step*batch_size:(step+1)*batch_size]

			# Perform the forward pass
			output = self.forward(batch_X, training=False)

			# Calculate the los
			self.loss.calculate(output, batch_y)

			# Get predictions and calculate an accuracy
			predictions = self.output_layer_activation.predictions(output)
			self.accuracy.calculate(predictions, batch_y)

		# Get and print validation loss and accuracy
		validation_loss = self.loss.calculate_accumulated()
		validation_accuracy = self.accuracy.calculate_accumulated()

		# Print a summary
		print(f'validation, ' +
			  f'acc: {validation_accuracy:.3f}, ' +
			  f'loss: {validation_loss:.3f}')

	# Predicts onthe samples
	def predict(self, X, *, batch_size=None):

		# Default value if batch size is not being set
		prediction_steps = 1

		# Calculate number of steps
		if batch_size is not None:
			prediction_steps = len(X) // batch_size
			# Dividing rounds down. If there are some remaining data,
			# but not a full batch, this won't include it 
			# Add '1' to include this not full batch
			if prediction_steps * batch_size < len(X):
				prediction_steps += 1

		# Model outputs
		output = []

		# Iterate over steps 
		for step in range(prediction_steps):

			# If batch size is not set  - train ussing one step and full dataset
			if batch_size is None:
				batch_X = X

			# Otherwise slice a batch 
			else:
				batch_X = X[step*batch_size:(step+1)*batch_size]

			# Perform the forward pass 
			batch_output = self.forward(batch_X, training=False)

			# Append batch prediciton to the list of predictions
			output.append(batch_output)

		# Stack and return results
		return np.vstack(output)

	# Retrieves and returns parameters of trainable layers
	def get_parameters(self):

		# Create a list for parameters
		parameters = []


		# Iterate trainable layers and get their parameters

		for layer in self.trainable_layers:
			parameters.append(layer.get_parameters())

		# Return a list
		return parameters

	# Updates the model with new parameters
	def set_parameters(self, parameters):

		# Iterate over the parameters and layers
		# and update each layers with each set of the parameters
		for parameters_set, layer in zip(parameters, self.trainable_layers):
			layer.set_parameters(*parameters_set)

	# Saves the parameters to a file
	def save_parameters(self, path):

		# Open a file in the binary-write mode and save parameters to it
		with open(path, 'wb') as f:
			pickle.dump(self.get_parameters(), f)

	# Load the weights and updates a model instance with them
	def load_parameters(self, path):

		# Open file in the binary-read mode, load weights and update trainable layers
		with open(path, 'rb') as f:
			self.set_parameters(pickle.load(f))

	# Saves the model
	def save(self, path):

		# Make a deep copy of current model instance
		model = copy.deepcopy(self)

		# Reset accumulated values in loss and accuracy objects
		model.loss.new_pass()
		model.accuracy.new_pass()

		# Remove data from input layer and gradients from the loss object
		model.input_layer.__dict__.pop('output', None)
		model.loss.__dict__.pop('dinputs', None)

		# For each layer remove inputs, output and dinputs properties
		for layer in model.layers:
			for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
				layer.__dict__.pop(property, None)

		# Open a file in the binary-write mode and save the model
		with open(path, 'wb') as f:
			pickle.dump(model, f)

	##-- Outputs stats about the model (Written by Aloïs Berger)
	def stats(self, sigma, path_name=None):

		# Print stats
		# Other layers
		lay = [self.loss, self.optimizer] # self.accuracy can be added
		le = len(self.layers) + len(lay)
		l = len(self.layers)


		# Set figure up
		fig = plt.figure(constrained_layout=True,figsize=(20,10))
		out_gs = fig.add_gridspec(2,7)

		# Get weights and biases
		weights = []
		biases = []
		for layer in self.trainable_layers:
			weights.append(layer.weights.tolist())
			biases.append(layer.biases.tolist())

		f_ax1 = fig.add_subplot(out_gs[0,0])
		f_ax1.set_title('Model Struktur:')
		layer_name = []
		for i in range(le):
			if i < l:
				f_ax1.text(0.1,1-(i+1.5)*(1/(le+1)),f'Layer{i}: {self.layers[i].stat}')
				layer_name.append(self.layers[i].stat)
			else:
				f_ax1.text(0.1,1-(i+1.5)*(1/(le+1)),f'Layer{i}: {lay[i-l].stat}')
				layer_name.append(lay[i-l].stat)

		f_ax1.set_axis_off()
		f_ax2 = fig.add_subplot(out_gs[0,2:4])
		f_ax2.plot([np.average(loss_list[i:i+sigma]) for i in range(len(loss_list))])
		f_ax2.set_title("Loss")
		f_ax2.set_xlabel("Steps")
		f_ax3 = fig.add_subplot(out_gs[1,0:2])
		f_ax3.plot([np.average(lr_list[i:i+sigma]) for i in range(len(lr_list))])
		f_ax3.set_title("Learning_rate")
		f_ax3.set_xlabel("Steps")
		f_ax4 = fig.add_subplot(out_gs[1,2:4])
		f_ax4.plot([np.average(accuracy_list[i:i+sigma]) for i in range(len(accuracy_list))])
		f_ax4.set_title("Accuracy")
		f_ax4.set_xlabel("Steps")

		f_ax5 = fig.add_subplot(out_gs[0,5])
		f_ax5.set_title("Weights")
		f_ax5.xaxis.set_visible(False)
		f_ax5.yaxis.set_visible(False)
		f_ax5.spines["left"].set_color("white")
		f_ax5.spines["right"].set_color("white")
		f_ax5.spines["bottom"].set_color("white")
		f_ax5.spines["top"].set_color("white")

		f_ax5_inner = out_gs[:2,5].subgridspec(1,len(weights))
		axs5 = f_ax5_inner.subplots(sharey=True)
		for a, ax5 in np.ndenumerate(axs5):				
			ax5.pcolormesh(np.arange(0,len(weights[a[0]][0])+1,1),np.arange(0,len(weights[a[0]])+1,1),
						  weights[a[0]], cmap=plt.get_cmap('seismic'))
			ax5.set_title(f"\n Layer {a[0]+1}")
			ax5.set_xlabel("Neuron")

		f_ax6 = fig.add_subplot(out_gs[0,6])
		f_ax6.set_title("Biases")
		f_ax6.xaxis.set_visible(False)
		f_ax6.yaxis.set_visible(False)
		f_ax6.spines["left"].set_color("white")
		f_ax6.spines["right"].set_color("white")
		f_ax6.spines["bottom"].set_color("white")
		f_ax6.spines["top"].set_color("white")

		f_ax6_inner = out_gs[:2,6].subgridspec(1,len(biases))
		axs6 = f_ax6_inner.subplots(sharey=True)
		for b, ax6 in np.ndenumerate(axs6):
			#norm = cl.Normalize(vmin=biases[b[0]].min(),vmax=biases[b[0]].max())			
			# pcm = ax6.pcolormesh(np.arange(0,2,1),np.arange(0,len(biases[b[0]][0])+1,1),
			# 			  np.array(biases[b[0]]).T, cmap=plt.get_cmap('seismic'), norm=norm)
			# fig.colorbar(pcm, ax=ax6, location="right")
			ax6.pcolormesh(np.arange(0,2,1),np.arange(0,len(biases[b[0]][0])+1,1),
			 			   np.array(biases[b[0]]).T, cmap=plt.get_cmap('seismic'))
			ax6.set_title(f"\n Layer {b[0]+1}")
			if b[0]==0:
				ax6.set_ylabel("Neuron")
		
		# Save in a folder
		os.mkdir(path_name)
		full_path_name = str(path_name) + '/'

		# Save the figure
		path_name_png = str(full_path_name) + 'figure.PNG'
		plt.savefig(path_name_png)
		plt.show()
		
		path_name_model = str(full_path_name) + 'Network.model'
		self.save(path_name_model)

		statistics = np.array([layer_name, loss_list, lr_list, accuracy_list, weights, biases])
		with open(str(full_path_name)+'weibia', 'wb') as f:
			pickle.dump(statistics, f)
		# --##
		

	# Loads and returns a model
	@staticmethod
	def load(path):

		# Open file in the binary-read mode, load a model
		with open(path, 'rb') as f:
			model = pickle.load(f)

		# Return a model
		return model


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

	# Scan all the directories and create a list of labels
	labels = os.listdir(os.path.join(path, dataset))

	# Create lists for the samples and labels
	X = []
	y = []

	# For each label folder 
	for label in labels:
		# And for each image in given folder
		for file in os.listdir(os.path.join(path, dataset, label)):
			# Read the image 
			image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_GRAYSCALE)

			dic[str(label)+str(dataset)] += 1

			# And append it and a label to the lists
			X.append(image)
			y.append(label)

	# Convert the data to proper numpy arrays and return 
	return np.array(X), np.array(y).astype('uint8') # say that y is int and not float

# MNIST dataset (train + test)
def create_data_mnist(path):

	# Load both sets seperately
	X, y = load_mnist_dataset('train', path)
	X_test, y_test = load_mnist_dataset('test', path)

	# And return all the data
	return X, y, X_test, y_test


##-- Create a list to store all Losses
loss_list = []
accuracy_list = []
lr_list = []

dic = { "0train": 0, "1train": 0, "2train": 0, "0test": 0, "1test": 0, "2test": 0}
# --##

# Create dataset and test dataset
X, y, X_test, y_test = create_data_mnist('RockPaperScissors/Augmented_Data_resized28')

print(dic)

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

test = [np.resize(cv2.imread('RockPaperScissors/Augmented_Data/train/0/img-0-0.jpg', cv2.IMREAD_GRAYSCALE), (28,28)),
		np.resize(cv2.imread('RockPaperScissors/Augmented_Data/train/0/img-0-1.jpg', cv2.IMREAD_GRAYSCALE), (28,28))]
test = np.array(test)
test = (test.reshape(test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

print('image')
print(test)
print(len(test))
print(len(test[0]))
# plt.imshow(image,cmap='gray')

print('X')
print(X)
# plt.imshow(X[0],cmap='gray')
plt.show()

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

##-- For debuging
# print(X[0]) 
# print(len(X))
# print(len(X[0]))
# print(X)
# --##

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(X.shape[1], 784))
model.add(Activation_ReLU())
model.add(Layer_Dense(784, 784))
model.add(Activation_ReLU())
model.add(Layer_Dense(784, 784))
model.add(Activation_Softmax())

# Set loss,optimizer and accuracy objects
model.set(
	loss=Loss_CategoricalCrossentropy(),
	optimizer=Optimizer_Adam(learning_rate=0.001,decay=2e-3),
	accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=1, batch_size=512, print_every=100)

## If you want to load model instead of training a new one
# model = Model.load('Optimizing_RPS/test/Network.model')

RPS_labels = {
	0: 'rock',
	1: 'paper', 
	2: 'scissors'
}
##-- Testing the model with 3 samples
# for i in range(3):
# 	image_data = cv2.imread(str(RPS_labels[i])+'.png', cv2.IMREAD_GRAYSCALE)
# 	image_data = cv2.resize(image_data, (28,28))

# 	plt.imshow(image_data)
# 	plt.show()
# 	image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
# 	confidences = model.predict(image_data)
# 	print(confidences)
# 	print(len(confidences[0]))
# 	predictions = model.output_layer_activation.predictions(confidences)
# 	prediction = RPS_labels[predictions[0]]
# 	print(prediction)

# Predict the class of the picture "paperingtest"
image_data = cv2.imread('paperingtest.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28,28))
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
confidences = model.predict(image_data)
print(confidences)

model.stats(20, 'Optimizing_RPS/IMAGEs')
# --##