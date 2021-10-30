## Regression

## code from NNFS
## My own comments are marked with ##
## My own code start with ##-- and ends with --##

## Regression is about determining a specific value based on an input (for example predicting temperature)

import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data
import numpy as np

nnfs.init() # make every device the same

X, y = sine_data()

plt.plot(X, y)
plt.show()

## We will need the common Loss Class as well as some others:

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

		#gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:

	# Forward pass
	def forward(self, inputs):
		#remember the inputs values
		self.inputs = inputs
		#calculate ouput value
		self.output = np.maximum(0, inputs)

	# Backward pass
	def backward(self, dvalues):
		# Since we need to modify the originaal variable let's make a copy of the values first
		self.dinputs = dvalues.copy()

		# Zero gradient where input values were nagative
		self.dinputs[self.inputs <= 0] = 0

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

# Adam
## Adaptive Momentum
## it is RMSProp but with momentum build in
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

## Regression is about determining a specific value based on an input (for example predicting temperature)
## Therefor we will use a linear_avtivation function for the output layer

# Linear activation
class Activation_Linear:

	# Forward pass
	def forward(self, inputs):
		# Just remember values
		self.inputs = inputs
		self.output = inputs

	# Backward pass
	def backward(self, dvalues):
		# The derivative is 1, 1 * dvalues = dvalues - the chain rule
		self.dinputs = dvalues.copy()

## Sure those steps are unnecessary but they bring clarity and in terms of copmutatuional time
## it does not change much

## Loss: We are no longer using classification labels, which means we can't use cross-entropy.
## The two main methods used to calculate error in regression are mean squared error (MSE) 
## and mean absolute error (MAE)

## Mean squared error: You square the diffrence between the predicted and true values of single outputs
## and average those squared values
## The idea is to penalize more harshly the further away we get from the intended target.
## Derivative: 1/J constant; inner is 0-1, outer is 2(y_true - y_predicted)**2-1
## => (-1) * (1/J) * 2*(y_true - y_predicted)**1 = -(2/J) * (y_true - y_predicted)

# Mean Squared Error (MSE) loss
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

## Mean absolute Error Loss: You take the absolute difference between the predicted and true values
## in a single output and average those absolute values = |y_true - y_predicted|
## This function penalize error linearly. It (produces sparser results and) is robust against outliers
## Which can be both advantageous and disadvantageous
## In reality L1 (MAE) loss is less frequently used then L2 (MSE) loss

## Derivative: 1/J is a constant; if y_predicted is smaller then y_true you have y_true - y_predicted
## which results in the derivative beeing 1
## if y_predicted is bigger then y_true you have y_predicted - y_true which results in the Derivative
## beeing -1
## if |y_true - y_predicted| is 0 there is no derivative (we will use 0 as it is on the right spot and
## and doesn't need to change it's function)

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

## Now we want a way of measuring performance. Though there are 2 problems
## 1st We only have single output (can't have a common prediction)
## 2nd We get float which makes it almost impossible to get exact prediction 
## (for example if the temperature is 24.45 Degrees Celsius and the Model predicts 24.40 it won't be true
## which means that the accuracy will be 0) (before we had like 95% of the outputs were right)

## First, we need a cap or some limit to which we want to be accurate
## For that we will calculate the standard deviation from the ground truth target and divide it by a number
## The larger the number the more strict we are (here 250)

accuracy_precision = np.std(y) / 250

## We will use this number as a measure of how much is allowed

## Here it's correct/accurate when the diffrence is smaller then that buffer on top
## predictions = activation2.outputs
## accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

## Regression Model Training
## put it all together 

# Create dataset
X, y = sine_data()

# Create Dense layer with 1 input feature and 64 output values
dense1 = Layer_Dense(1, 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 64 inputs features (as we take output of previous layer here)
# and 1 output value
dense2 = Layer_Dense(64, 1)

# Create Linear activation
activation2 = Activation_linear()

# Create loss function
loss_function = Loss_MeanSquaredError()

# Create Optimizer
optimizer = Optimizer_Adam()

## Accuracy precision for accuracy calculation
## There are no really accuracy factor for regression problem, but we can simulate/approximate it.
## We'll calculate it by checking how many values have a diffrence to their ground truth equivalent
## less than given precision
## We'll calculate this precision as a fraction fo standard deviation of all the ground truth values
accuracy_precision = np.std(y) / 250

# Train in loop
for epoch in range(10001):

	# Perform a forward pass of our training data through this layer
	dense1.forward(X)

	# Perform a forward pass through activation function
	# takes the output of first dense layer here
	activation1.forward(dense1.output)

	# Perform a forward pass through second Dense layer
	# takes outputs of activation function of first layer
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

	# Calculate accuracy from output of activation2 and targets
	# To calculate it we're taking absolute differences between predictions and ground truth values
	# and compare if diffrences are lower than given precision value
	predictions = activation2.output
	accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)

	if not epoch % 100:
		print(f'epoch: {epoch}, ' +
			  f'acc: {accuracy:.3f}, ' +
			  f'loss: {loss:.3f} (' +
			  f'data_loss: {data_loss:.3f}, ' +
			  f'reg_loss: {regularization_loss:.3f}), ' +
			  f'lr: {optimizer.current_learning_rate}')

	# Backward pass
	loss_function.backward(activation2.output, y)
	activation2.backward(loss_function.dinputs)
	dense2.backward(activation2.dinputs)
	activation1.backward(dense2.dinputs)
	dense1.backward(activation1.dinputs)

	# Update weights and biases
	optimizer.pre_update_params()
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.post_update_params()

## It did pretty poorly. 
## Lets add an ability to draw the tesing data and let's also do a forward pass
## on the testing data drawing output data on the same plot as well

X_test, y_test = sine_data()

dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation2.output)
plt.show()

## The problem here is that we used a linear Activation function (pretty much doesn't do much)
## and to map nonlinear functions we there for needed at least 2 or more hidden layers 
## in this case we only have one + output layer and this isn't enough

## adjust params in the Regression full code

## Using 3 Layer helped but it seems stuck at a local minimum
## Therefor we will use a higher learning rate but add a learning rate decay
optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-3)

## It still got stuck but when I put it on 0.02 it worked pretty well( general shape down)
## By augmenting the decay to 2e-3 the curve was pretty much matched
## After some tweaks we found out that 0.005 learning rate was right on the dot

## Debugging such things is usually pretty hard

## In Chapter 3 we discussed parameter initialization mehtods (we set the weight to 0.01 
## but if we turn it to 0.1 the model learns way better)

## We can learn a imporant lesson here by looking at Keras 2 Library code and how they initialize weight
## They use Glorot uniform which depends on the number of neurons and is not constat like in our case

## We redo every model but with weight 0.1 this time around
## (Jumping accurarcy due to to high learning rate but in this case with weights 0.1 it still worked)
## This time all diffrent learning rate let the model to learn
## this shows how important initial weights are