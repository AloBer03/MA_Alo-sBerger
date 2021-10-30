## Loss functions

## code from NNFS
## My own comments are marked with ##
## My own code start with ##-- and ends with --##

import math
import numpy as np

# Categorical cross-entropy
# Possible with softmax output
softmax_output = [0.7, 0.1, 0.2]
# Ground truth 
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] + 
		 math.log(softmax_output[1])*target_output[1] + 
		 math.log(softmax_output[2])*target_output[2])

# Simplyfies into 
loss = - math.log(softmax_output[0])

# Instead of saying which out is right we just say what class is right (class 1 beeing 0, class 2 beeing 1...)
# We get some batch of inputs ## (class: 0=dog, 1=cat, 2=human)
softmax_outputs = [[0.7, 0.1, 0.2],
				  [0.1, 0.5, 0.4],
				  [0.02, 0.9, 0.08]]
class_targets = [0, 1, 1]			## now we want dog, cat, cat for our 3 batches

# for targ_idx, distribution in zip(class_targets, softmax_outputs):
#	print(distribution[targ_idx]) ## Print(the value of the class it should have been)

# Now with a numpy array
softmax_outputs = np.array([[0.7, 0.1, 0.2],
				  		    [0.1, 0.5, 0.4],
				  			[0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]			## Now we want dog, cat, cat for our 3 batches

print(softmax_outputs[[0, 1, 2], class_targets])
## Numpy let's us index array in multiple ways (here with a list)
## Here the first list tells us we want raw 0 then 1 then 2
## and the second list tells us wich column(value) we want from the respective raw 
## [raw:0 column:0  raw:1 column:1  raw:2 column:1] 
# Instead of having a list [0,1,2,...,n-1] we can write range(len(softmax_outputs)
print(softmax_outputs[range(len(softmax_outputs)), class_targets])
# Now add log
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
# Take an avergage of a batch with numpy
average_loss = np.mean(neg_log)

# Though it only works if there is only one hot value
print(average_loss)


# When more use the full function

softmax_outputs = np.array([[0.7, 0.1, 0.2],
				  		    [0.1, 0.5, 0.4],
				  			[0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],		# Dog
						  [0, 1, 0],		# Cat
						  [0, 1, 0]])		# Cat

if len(class_targets.shape) == 1: # Only one hot value
	correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]
elif len(class_targets.shape) == 2:
	correct_confidences = np.sum(softmax_outputs*class_targets, axis=1)

# Loss
neg_log = -np.log(correct_confidences)
# Average
average_loss = np.mean(neg_log)

print(average_loss)

# Last problem is log(0) which equals to infinity.
# So we clip at both side of the by a very small number (S.122)
y_pred = [1,2,3]
y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7)

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

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_outputs, class_targets)
print(loss)

# Calculate accuracy

# Probabilities of 3 samples (#output of activation2)
softmax_outputs = np.array([[0.7, 0.2, 0.1],
						   [0.5, 0.1, 0.4],
						   [0.02, 0.9, 0.08]])
# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])

# Calculate values along second axis (index:1) (here horizonally)
predictions = np.argmax(softmax_outputs, axis=1)
# If targets are one-hot encoded - convert them
if len(class_targets.shape) == 2:
	class_targets = np.argmax(class_targets, axis=1) # argmax?
# True evaluates to 1; False to 0 (?)
accuracy = np.mean(predictions == class_targets)

print('acc:', accuracy)


## Regularization
## add the sum(Squared) of weights and biases to the loss
## This should ensure that there will be no weights/biases with high values( exploding gradient = insability)
## and that all neurons participate
## Lambda helps dictate how much impact the regularization has

## l1w = lambda_l1w * sum(abs(weights))
## l1b = lambda_l1b * sum(abs(biases))
## l2w = lambda_l2w * sum(weights**2)
## l2b = lambda_l2b * sum(biases**2)
## loss = data_loss + l1w + l1b + l2w + l2b

## (because you changed the loss you have to change the loss backpass (gradient))

# Backwards pass L1
weights = [0.2, 0.8, -0.5] # Weights of one neurons
dL1 = [] # Array of partial derivatives of L1 Regularization
for weight in weights:
	if weight >= 0:
		dL1.append(1)
	else:
		dL1.append(-1)

# Now Multiple Neurons
weights = [[0.2, 0.8, -0.5, 1],		# Now we have 3 sets of weights = 3 neuron 
		   [0.5, -0.91, 0.26, -0.5],
		   [-0.26, -0.27, 0.12, 0.87]]
dL1 = [] # Array of partial derivatives of L1 regularization
for neuron in weights:
	neuron_dL1 = []	# Derivatives related to one neuron
	for weight in neuron:
		if weight >= 0:
			neuron_dL1.append(1)
		else:
			neuron_dL1.append(-1)
	dL1.append(neuron_dL1)

print(dL1)

# For Numpy version we fill a array (with the shape of weights (np.ones_like()) 
# and where weights is smaller than 0 we put -1

weights = np.array([[0.2, 0.8, -0.5, 1], 
		   			[0.5, -0.91, 0.26, -0.5],
		   			[-0.26, -0.27, 0.12, 0.87]])

dL1 = np.ones_like(weights)

dL1[weights < 0] = -1

print(dL1)

## Update the backwards pass 
## For L1 we take dL1 and multilply it with lambda for both weights and biases | lambda* (x > 0 => 1 
##																					     x < 0 => -1)
## For L2 we take the weights/biases and multiply it with 2*lambda | lambda* (x**2 => 2*x)
## Calculate regularization penalty



## Regularization_loss = loss_function.regularization_loss(dense1) + \
##					  loss_function.regularization_loss(dense2)

## Calculate overall loss
## Loss = data_loss + regularization_loss