# Testing

## code from NNFS
## My own comments are marked with ##
## My own code start with ##-- and ends with --##

import Neural_Network_modified_Alois as nna
from matplotlib import pyplot as plt
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import nnfs 
import numpy as np
import pickle
from timeit import timeit

nnfs.init()

a = np.random.randn(2,4)
b = np.zeros((2,2))
print(a)
print(b)


n_inputs = 2
n_neuron = 4
weights = 0.01 * np.random.randn(n_inputs, n_neuron) 
biases = np.zeros((1, n_neuron))
print(weights, biases)

X, y = spiral_data(samples=100, classes=3) # spiral_data or vertical_data
# plt.scatter(X[:,0], X[:,1], c=y, cmap= 'brg')
# plt.show()

# Test random manipulating weights and biases
# Class of a fully connected layer (dense Layer)
class Layer_Dense:
	# Initialize (for ex. for preloaded model) here randomly
	def __init__(self, n_inputs, n_neuron):
		# Initialize weights and bias
		self.weights = 0.01 * np.random.randn(n_inputs, n_neuron) # randomly
		# Note inputs then neuron  (not Neuron then input) so we don't need to transpose afterwards
		self.biases = np.zeros((1, n_neuron)) #all zero (can be diffrent)

	# Forward pass (passing data from one layer to the other)
	def forward(self, inputs):
		# Calculate output through input, weights, bias
		self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:

	# Forward pass
	def forward(self, inputs):
		# Calculate ouput value
		self.output = np.maximum(0, inputs)

class Activation_Softmax:

	# Forward pass
	def forward(self, inputs):

		# Get unnormalized probability	here substracting highest value		  
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True)) 
																			  
		# Normalize them for each sample 
		# Numpy divides each value from each outputbatch by the corresponding row value  # [[sum of batch 1],
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)			 #= [sum of batch 2],
																						 #  [sum of n-batch]]
		self.output = probabilities

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

# Create model
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
# Loss
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 9999999 
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(1):
	# Generate a new set of weights for iteration
	dense1.weights += 0.05 * np.random.randn(2, 3) # += to adjust and not completly randomize
	dense1.biases += 0.05 * np.random.randn(1, 3)
	dense2.weights += 0.05 * np.random.randn(3, 3)
	dense2.biases += 0.05 * np.random.randn(1, 3)

	# Perform a forward pass of the training data through this layer
	dense1.forward(X)
	activation1.forward(dense1.output)
	dense2.forward(activation1.output)
	activation2.forward(dense2.output)

	# Loss
	loss = loss_function.calculate(activation2.output, y)

	# Accuracy
	predictions = np.argmax(activation2.output, axis=1)
	accuracy = np.mean(predictions == y)

	# If loss is smaller -  print and save weights and biases aside
	if loss < lowest_loss:
		print('New set of weights found, iteration:', iteration, 'loss;', loss, 'acc:', accuracy)
		best_dense1_weights = dense1.weights.copy()
		best_dense1_biases = dense1.biases.copy()
		best_dense2_weights = dense2.weights.copy()
		best_dense2_biases = dense2.biases.copy()
		lowest_loss = loss
	# Revers weights and biases (add)
	else:
		dense1.weights = best_dense1_weights.copy()
		dense1.biases = best_dense1_biases.copy()
		dense2.weights = best_dense2_weights.copy()
		dense2.biases = best_dense2_biases.copy()

# With veritcal data we got significant improvement
# With spiral data we only got a small bump in accuracy and loss

# Transform one-hot encoded vector into true-labels
y_true= np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
print(np.argmax(y_true, axis = 1))

# Test avtivation Softmax_Loss_CategoricalCrossentropy():
# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backwards step
class Activation_Softmax_Loss_CategoricalCrossentropy():

	# Creates activation and loss function objects
	def __init__(self):
		self.activation = Activation_Softmax() 
		self.loss = Loss_CategoricalCrossentropy()

	# Froward pass
	def forward(self, inputs, y_true):
		# Output layer's activation function
		self.activation.forward(input)
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
		#calculate gradient
		self.dinputs[range(samples), y_true] -= 1.0
		#normalize gradient
		self.dinputs = self.dinputs / samples


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

		#losses
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

##--
softmax_outputs = np.array([[0.7, 0.1, 0.2],
						   [0.1, 0.5, 0.4],
						   [0.02, 0.9, 0.08]])

class_targets = np.array([0,1,1])

softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_CategoricalCrossentropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs
print('Gradient: combined loss and activation:', dvalues1)
print('Gradient: seperate loss and activation:', dvalues2)
# Speed test
def f1():
	softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
	softmax_loss.backward(softmax_outputs, class_targets)
	dvalues1 = softmax_loss.dinputs

def f2():
	activation = Activation_Softmax()
	activation.output = softmax_outputs
	loss = Loss_CategoricalCrossentropy()
	loss.backward(softmax_outputs, class_targets)
	activation.backward(loss.dinputs)
	dvalues2 = activation.dinputs

t1 = timeit(lambda: f1(), number=10000)
t2 = timeit(lambda: f2(), number=10000)
print(t2/t1)
# Around 4 times faster !!
x = np.array([1])
path = 'Optimizing_RPS/model'
with open(path, 'wb') as f:
			pickle.dump(x, f)
z = np.array([[-0.28511015, 0.085136, -0.23976544, 0.0332373 , -0.04783594,
        -0.00474767, -0.36151865,  0.09533457, -0.24136847,  0.19265246],
       [ 0.01212846,  0.14249018,  0.08595020,  0.06509143,  0.03300954,
        -0.02794621,  0.01757961, -0.06218105, -0.18198262, -0.28572765],
       [-0.10104780, -0.23965563, -0.25808346,  0.00890746, -0.00371543,
         0.04030991, -0.01197709,  0.11664747,  0.00711345,  0.07309830],
       [ 0.06387720,  0.03875627,  0.11981798,  0.00209285,  0.07872984,
        -0.15345243,  0.07333111, -0.29223657,  0.03764989, -0.34737630],
       [ 0.02820562, -0.04138929, -0.02049194, -0.09171063, -0.00896618,
         0.02865623, -0.03595628, -0.00383713, -0.03656452, -0.03101091],
       [ 0.14495598,  0.07926853, -0.09214851,  0.09436148, -0.12581060,
        -0.00808079,  0.03891787, -0.51078400,  0.00521480,  0.03779042],
       [-0.00215332,  0.00888970, -0.00344901, -0.01331208, -0.02354234,
        -0.00678951, -0.00181804,  0.00618549, -0.00133857,  0.00114086],
       [-0.05432996, -0.16465087, -0.13396642, -0.18570468, -0.30843544,
         0.10941016, -0.14372893,  0.14177097, -0.03474295,  0.06480630],
       [-0.28603598, -0.08495274, -0.12360594, -0.08509547,  0.11039869,
         0.05785963, -0.10229932,  0.10839801,  0.08872057,  0.02393923],
       [ 0.08114810,  0.10316103,  0.03766327,  0.04889860, -0.04138971,
         0.00679970,  0.02724111, -0.14349583, -0.13861856, -0.23839633],
       [-0.04702927, -0.43679094, -0.02544295, -0.05649933,  0.04199936,
        -0.09388859,  0.05779373, -0.01037198,  0.21566187,  0.06977680]])

z = np.reshape(z,-1)
t  = np.array([[-0.1,-2,-3],[0.2,-0.2,0],[1,2,3]])
x = np.arange(0, len(z)+1, 1)  
y = np.arange(0, 2, 1)  

z = [z]
print("X", x)
print("Y", y)
print("Z", z)


fig, ax = plt.subplots()
ax.pcolormesh(x, y, z, cmap=plt.get_cmap('seismic'))
plt.show()
print("X", x)
print("Y", y)
print("Z", z)

print(nna.model)

# --##