#classes
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


#class of a fully connected layer (dense Layer)
class Layer_Dense:

	#initialize (for ex. for preloaded model) here randomly
	def __init__(self, n_inputs, n_neuron):
		#initialize weights and bias
		self.weights = 0.01 * np.random.randn(n_inputs, n_neuron) # randomly
		#note inputs then neuron  (not Neuron then input) so we don't need to transpose afterwards
		self.biases = np.zeros((1, n_neuron)) #all zero (can be diffrent)

	# forward pass (passing data from one layer to the other)
	def forward(self, inputs):
		#remember input values
		self.inputs = inputs
		#calculate output through input, weights, bias
		self.output = np.dot(inputs, self.weights) + self.biases

	#Backward pass
	def backward(self, dvalues):
		#gradient on parameters
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=0)
		#gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:

	#forward pass
	def forward(self, inputs):
		#remember the inputs values
		self.inputs = inputs
		#calculate ouput value
		self.output = np.maximum(0, inputs)

	#backward pass
	def backward(self, dvalues):
		#since we need to modify the originaal variable let's make a copy of the values first
		self.dinputs = dvalues.copy()

		#zero gradient where input values were nagative
		self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

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
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
			#calculate sample-wise gradient and add it to the array of samples gradients
			self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


#common loss class 
class Loss: 

	#calculates the data and regularization losses
	#given model output and ground truth values
	def calculate(self, output, y):

		#calculate sample losses
		sample_losses = self.forward(output, y)

		#calculate mean loss
		data_loss = np.mean(sample_losses)

		#return loss
		return data_loss

class Loss_CategoricalCrossentropy(Loss):

	#forward pass
	def forward(self, y_pred, y_true):
		#number of samples
		samples =  len(y_pred)

		#clip data to prevent division by 0
		#clip both sides to not drag mean towards any value
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		#probabilities for target values only if categorical labels (one dimension [dog,cat,cat])
		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]

		#mask values - only for one-hot encoded labels (multiple dimension [[1,0,0], [0,1,0], [0,1,0]])
		elif len(y_true.shape) == 2:
			correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

		#losses
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

	#backwards pass
	def backward(self, dvalues, y_true):

		#number of samples
		samples = len(dvalues)
		#number of labels in ervery sample (use first sample to count them)
		labels = len(dvalues[0])

		#if labels are sparse, turn them into one-hot vector
		if len(y_true.shape) ==1:
			y_true = np.eye(labels)[y_true]

		#calculate gradient
		self.dinputs = -y_true / dvalues
		#normalize gradient
		self.dinputs = self.dinputs / samples

#softmax classifier - combined Softmax activation and cross-entropy loss for faster backwards step
class Activation_Softmax_Loss_CategoricalCrossentropy():

	#creates activation and loss function objects
	def __init__(self):
		self.activation = Activation_Softmax() 
		self.loss = Loss_CategoricalCrossentropy()

	# Froward pass
	def forward(self, inputs, y_true):
		#output layer's activation function
		self.activation.forward(inputs)
		#set output
		self.output = self.activation.output
		#calculate and return the loss value
		return self.loss.calculate(self.output, y_true)

	#Backwards pass
	def backward(self, dvalues, y_true):

		# Number of samples
		samples = len(dvalues)
		# If labels are one-hot encoded turn them into discrete values
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)

		# Copy so we can sagely modify 
		self.dinputs = dvalues.copy()
		#calculate gradient
		self.dinputs[range(samples), y_true] -= 1
		#normalize gradient
		self.dinputs = self.dinputs / samples


#creating samples
X, y = spiral_data(samples= 100, classes = 3) # X is input and y is output (if given x it should give y out)

#create object (dense layer 2 inputs and 3 neurons)
dense1 = Layer_Dense(2, 3)

#create activatoin ReLU (to use afterwards)
activation1 = Activation_ReLU()

#creating second dense layer
dense2 = Layer_Dense(3, 3) #3 inputs and 3 Neuron (here then ouputs)

#create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#perform a forward pass (calculating output)
dense1.forward(X)

#pass ouput of previous layer through activation function
activation1.forward(dense1.output)

#taking the ouput of the previous layer and pass it through weights and bias
dense2.forward(activation1.output)

#Perform a forward pass through the activation/loss function 
#takes the output o second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)

#let's see output of the first few samples: 
print(loss_activation.output[:5])

#print loss values
print('loss:', loss)

#calculate accuracy from outputs of activation2 and targets
#calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
	y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

#print accuracy
print('acc:', accuracy)

# Backwards pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

#print(gradients)
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)



