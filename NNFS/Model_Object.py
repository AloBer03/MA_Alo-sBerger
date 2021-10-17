# Model_Object

import numpy as np
import nnfs
from nnfs.datasets import sine_data
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

		#gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)

# Dropout
class Layer_Dropout:

	# Init
	def __init__(self, rate):
		# Store rate, we invert it as for example for dropout of 0.1 we need success rate of 0.9
		self.rate = 1 - rate

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


class Activation_ReLU:

	# Forward pass
	def forward(self, inputs, training):
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

	# Calculate predictions for outputs
	# even though we will never use this for output layer (just for completeness)
	def predictions(self, outputs):
		return outputs

class Activation_Softmax:

	# Forward pass
	def forward(self, inputs, training):

		# Remember inputs value
		self.inputs = inputs

		# Get unnormalized probability	here substracting highest value		  
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
																			  
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

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return np.argmax(outputs, axis=1)

# Sigmoid activation
class Activation_Sigmoid:

	# Forward pass
	def forward(self, inputs, training):
		# Save input and calculate/save output of the sigmoid function
		self.inputs = inputs
		self.output = 1 / (1 + np.exp(-inputs))

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

	# Backward pass
	def backward(self, dvalues):
		# The derivative is 1, 1 * dvalues = dvalues - the chain rule
		self.dinputs = dvalues.copy()

	# Calculate predictions for outputs
	def predictions(self, outputs):
		return outputs

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

	# Set/remember trainable layers
	def remember_trainable_layers(self, trainable_layers):
		self.trainable_layers = trainable_layers

	# Calculates the data and regularization losses
	# Given model output and ground truth values
	def calculate(self, output, y, *, include_regularization=False):

		# Calculate sample losses
		sample_losses = self.forward(output, y)

		# Calculate mean loss
		data_loss = np.mean(sample_losses)

		# If just data loss - return it
		if not include_regularization:
			return data_loss

		# Return loss
		return data_loss, self.regularization_loss()

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
			correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

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
		if len(y_true.shape) == 1:
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

class Loss_MeanSquaredError(Loss): # L2 loss
	
	# Forward pass
	def forward(self, y_pred, y_true):

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
class Loss_MeanAbsoluteError(Loss): # L1 Loss
	
	# Forward pass
	def forward(self, y_pred, y_true):

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
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

	# Update parameters 
	def update_params(self, layer):
		# Momentum
		# If we use momentum
		if self.momentum:

			# If layer does not contain momentum arrays, create them filled with zeros
			if not hasattr(layer, 'weight_momentums'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				#if there is no momentum array for weights
				# THe array doesn't exist for biases yet either
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
		

	# Call once before any parameter updates 
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
# Adaptive Momentum is RMSProp but with momentum build in
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

	# Set loss and optimizer 
	def set(self, *, loss, optimizer, accuracy): 
		# The asterisk in the parameter requires keyword arguments for the following parameters
		# That way you have to pass in the names and values, which makes the code more legible
		self.loss = loss
		self.optimizer = optimizer
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

			# All layers except for the first and last
			elif i < layer_count -1:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.layers[i+1]

			# The last layer - the next object is the loss
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
		self.loss.remember_trainable_layers(self.trainable_layers)

		# If output activation is Softmax and loss function is Categorical Cross-Entropy
		# create an object of combined activation and loss function containing
		# faster gradient calculation
		if isinstance(self.layers[-1], Activation_Softmax) and \
		   isinstance(self.loss, Loss_CategoricalCrossentropy):
			# Create an object of combined activation and loss function
			self.softmax_classifier_output = \
				Activation_Softmax_Loss_CategoricalCrossentropy()

	# Train the model
	def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

		# Initialize accuracy object
		self.accuracy.init(y)


		# Main training loop
		for epoch in range(1, epochs+1):

			# Perform the forward pass
			output = self.forward(X, training=True)

			# Calculate loss
			data_loss, regularization_loss = self.loss.calculate(output, y, 
																 include_regularization=True)
			loss = data_loss + regularization_loss

			# Get predictions and calculate an accuracy
			predictions = self.output_layer_activation.predictions(output)
			accuracy = self.accuracy.calculate(predictions, y)

			# Perform backward pass
			self.backward(output, y)

			# Optimize (update parameters)
			self.optimizer.pre_update_params()
			for layer in self.trainable_layers:
				self.optimizer.update_params(layer)
			self.optimizer.post_update_params()

			# Print a summary
			if not epoch % print_every:
				print(f'epoch: {epoch}, ' +
					  f'acc: {accuracy:.3f}, ' +
					  f'loss: {loss:.3f} (' +
					  f'data_loss: {data_loss:.3f}, ' +
					  f'reg_loss: {regularization_loss:.3f}), ' +
					  f'lr: {self.optimizer.current_learning_rate}')

		# If there is the validation data
		if validation_data is not None:

			# For better readability
			X_val, y_val = validation_data

			# Perform the forward pass
			output = self.forward(X_val, training=False)

			# Calculate the loss
			loss = self.loss.calculate(output, y_val)

			# Get predictions and calculate an accuracy
			predictions = self.output_layer_activation.predictions(output)
			accuracy = self.accuracy.calculate(predictions, y_val)

			# Print a summary 
			print(f'validation, ' + 
				  f'acc: {accuracy:.3f}, ' + 
				  f'loss: {loss:.3f}')

	# Forward pass (use for training and validation (also called model inference))
	def forward(self, X, training):

		# Call forward method on the input layer this will set the output property that
		# the first layer in "prev" object as a parameter
		self.input_layer.forward(X, training)

		# Call forward method of every object in a chain 
		# Pass output of the previous object as a parameter
		for layer in self.layers:
			layer.forward(layer.prev.output, training)

		# "layer" is now the last object from the list
		# return its output
		return layer.output

	# Performs backwards pass
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
			for layer in reversed(self.layers[:-1]): # all to the last one (last one not included)
				layer.backward(layer.next.dinputs)

			return

		# First call backward method on the loss this will set dinputs property
		# that the last layer will try to access shortly
		self.loss.backward(output, y)

		# Call backward method going through all the objects in reversed order
		# passing dipunpts as a parameter
		for layer in reversed(self.layers):
			layer.backward(layer.next.dinputs)

# Training Problem: We would do a loop over all taking the previous Layer's output
# but the first layer has no previous output => we create a input layer with no weights and biases
# We create a Layer_input (similar to Layer_Dense)
# Input "layer"
class Layer_Input:

	# Forward pass
	def forward(self, inputs, training):
		self.output = inputs

	# Though no need of backward pass

# Common accuracy class
class Accuracy:

	# Calculate an accuracy
	# given predictions and ground truth values
	def calculate(self, predictions, y):

		# Get comparison results
		comparisons = self.compare(predictions, y)

		# Calculate an accuracy
		accuracy = np.mean(comparisons)

		# Return accuracy
		return accuracy

# Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):

	def __init__(self):
		# Create precision property
		self.precision = None

	# Calculates precision property based on passed-in ground truth
	def init(self, y, reinit=False):
		if self.precision is None or reinit:
			self.precision = np.std(y) / 250

	# Compares predictions to the ground truth values
	def compare(self, predictions, y):
		return np.absolute(predictions - y) < self.precision

# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):

	def __init__(self, *, binary=False):
		# Binary model?
		self.binary = binary

	# No initialization is needed
	def init(self, y):
		# Needs to exist because it's called automatically
		pass

	# Compares predictions to the ground truth values
	def compare(self, predictions, y):
		if not self.binary and len(y.shape) == 2:
			y = np.argmax(y, axis=1)
		return predictions == y

# Create dataset and test dataset
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# For Binary logistic Regression:
# Reshape labels to be a list of lists: Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case:
#  y = y.reshape(-1, 1)
#  y_test = y_test.reshape(-1, 1)
# We add L2 and we can delete one layer (because we have activation twice (sigmoid instead of linear))
# Change loss from MeanSquareError to BinaryCrossentropy
# Change optimizer parameters (learningrate form 0.005 to default, decay from 1e-3 to 5e-7)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss and optimizer objects
model.set(
	loss=Loss_CategoricalCrossentropy(),
	optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
	accuracy=Accuracy_Categorical())


# Finalize the model
model.finalize()

# Train model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)

# For backwards pass we make a function called trainable layers to remember the trainable layers
# in the Loss class and with that we can make the Regularization iterable over all the trainable layers

# Next we need to calculate the accuracy. First we need the predictions, which varied 
# depending on the activation function we were using
# We want to try to make the model use the right one without us telling it

# For this we'll add a predictions method to each activation function
# and we will call the predictions method later in the finalize function

# Now we need to calculate accuracy. For that we will make a common accuracy class (similar to loss)

# With that we made our model but it has not handle every type.

# Next will implement binary logistic regression. For this, we need to add two things.
# First, we need to calculate the categorical accuracy: (class on top)
# Next we need to validate the model (requires only a forward pass and a loss calculation)
# We modify the calculate method in Loss so it can calculate loss for validation
# We then also change the train model to do validation 

# Now let's test it (change back to spiral_data_set)

# Now we implement the Dropout_layer again
# For that we put in every Layer and activation a training parameter which tells us if we do 
# dropout or not ( because for validation we don't won't dropout to be happening)

# Next problem is the Softmax_Activation which we can't do anymore becaus it's in a loop
# (Before we did it manually therefor we could just write it instead of both other)
# First we want it to recognize if it is a classifier (Softmax and Categorical Cross-Entropy loss)
# We achieve that by looking at the last layer's and they are Softmax and CategoricalCrossEntropy
# we merge them together. We add this all to the finalize method of the model
# We merge them together by creating a new object (not in layers)
# For the backward pass (We just calculate dinputs and put them into the last layer's dinputs)
# (last layer beeing activation and not Loss)
# Then you initiate the backward pass except for the last layer(beeing activation layer)
# Because he has dinputs already set (last activation and loss already done)
