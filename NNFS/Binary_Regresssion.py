# Binary Regression

# Instead of saying how confident it is, that it is this class it says if it's yes or no.
# It could be anything from cat vs dog or cat vs not cat...

# With the Softmax it was exponential  

# Though for regression we will use the Sigmoid activation Function which squizes the values
# between 1 and 0 

import numpy as np

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

# Binary Cross-Entropy Loss

# Same negative log concept as Categorical Cross-Entropy loss
# but instead of only calculating it on the target class
# (Because classification wants to know which class)
# we will sum the log-likelihoods of the correct and incorrect class for each neuron
# Regression (which mean you have to "how much that class is" (if it's true of false / -1 or 1 (but not which)))

sample_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Now we need a sample loss for all the outputs

sample_losses = np.mean(sample_losses, axis=-1) # axis -1 = the last axis

# Explained
outputs = np.array([[1, 2, 3],
			    	[2, 4, 6],
			    	[0, 5, 10],
			    	[11, 12, 13],
			    	[5, 10, 15]])
print(np.mean(outputs, axis = -1))

# Now we have the sample and we inherit the calculate function from the Loss class
# which will calculate for us

# For the backwards pass we calculate the Derivative of the Binary Cross-Entropy but also
# the derivative of the Loss (sample with respect of each individual output) L i of l i,j
# We already have the second part ( derivative of single output loss with respect to the related prediction)
# (We add all those up and then divide by j (number of outputs))
# (Now we need to calculated the derivative of the samples loss)
# and now combine them

# Number of samples
samples = len(dvalues)
# Number of outputs in every sample
# WE'll use the first sample to count them
outputs = len(dvalues[0])

# Calculate gradient
self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs # j


# Normalize gradient
self.dinputs = self.dinputs / samples

# A further problem is that np.log is calculated such a way that log(0) equals - infinity
# There for we we clipp the batch of values so we are sure that there is no 0 in it

# Test print(np.log(0)) => error

# Clip data to prevent dicision by 0
# Clip both sides to not drag mean towards any value
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

# Calculate samle-wise loss
sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

# Other problem occours: During derivative calculation gradient can be 1 or 0
#  which for (1 - y_true) / (1 - dvalues) or y_true / dvalues (because you get division by 0)
# Therefor we will clip them as well

# Clip data to prevent division by 0
# Clip both sides to not drag mean towards any value
clipped_dvalues = np.clip(dvalues, 1e-7, 1e-7)

# Calculate gradient
self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

# Full Code

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
		clipped_dvalues = np.clip(dvalues, 1e-7, 1e-7)

		# Calculate gradient
		self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
		# Normalize gradient
		self.dinputs = self.dinputs / samples


# What's left is to tweek some things to implement it into our Network:

# - Change the spiral_data to 2 classes
# We reshape our labels as they're not sparse anymore. They are Binary, 0 or 1:
# From [0, 0, 0, 0] to [[0],[0],[0],[0]] 
# Not 100§ clear yet p.405

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)

# We also have to change the !!!!!!! p.405