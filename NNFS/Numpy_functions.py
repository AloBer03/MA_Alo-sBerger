# Numpy_functions

# (Own code)
# Why use Numpy? 
# Because it's quicker (python libraries can be writen in C which can be much faster)
# It also makes the code much cleaner
# I still tried to recreate some Numpy function

def randn(x, y):
	ret = []
	for i in range(x):
		ret.append([np.random.randn() for y in range(y)])
	return ret


# Dotproduct vector
def dotproduct(input_value, weights):
	#only vectors 
	output = 0
	for a, b in zip(input_value, weights):
		output += a*b
	return output

# Transposition
def transposition(matrix):
	transposed_matrix = [[] for x in range(len(matrix[0]))]
	for j in range(len(matrix)):
		for i in range(len(matrix[j])):
			transposed_matrix[i].append(matrix[j][i])
	return transposed_matrix

# Dotproduct with matrices
def dotmatrix(matrix1, matrix2):
	output_matrix = [[] for x in range(len(matrix1))]
	for i in range(len(matrix1)):
		additionar = 0
		for j in range(len(matrix1[0])):
			additionar += matrix1[i][j] * matrix2[j][i]
		output_matrix[i].append(additionar)
	return output_matrix

# Adding vector
def addition(vec1, vec2):
	output = []
	for a, b in zip(vec1, vec2):
		output.append(a+b)
	return(output)

# Random (np.random.randn())
# Normal (gaussian) distribution with mean 0 and variance (1)
# and * 0.01 small enough to not affect training???
# Parameters are the dimension of the output matrix

# np.zeros() creates a matrix with the dimension being the parameters
# and all values being zero

# Numpy maximum (ReLU avtivation function)
def activeate_ReLU(bias, inputs):
	output = []
	for i in inputs:
		if i > -bias:
			output.append(i+bias)
		else:
			output.append(0.0)

# Sum function (axis=1)
def sums(inputs,axis=1,keepdims=True):
	output = []
	for i in inputs:
		output.append([sum(i)])
	return output

# Highest value in a matrix (list wise)
def argmax(inputs):
	output = []
	for batch_set in inputs:
		previous_value = batch_set[0]
		for value in batch_set:
			if value > previous_value:
				previous_value = value
			else:
				pass
		output.append([previous_value])
	return output

# Average (mean)
def mean(inputs):
	output = sum(inputs) / len(inputs)
	return output

# Clipping on both side
def clipped(inputs, lower_boundery, upper_boundery):
	while inputs[0] < lower_boundery: 
		del inputs[0]
	while inputs[-1] > upper_boundery:
		del inputs[-1]
	return inputs

# Creating a list/array with steps
def arrange(start, end, step):
	length = (end - start) /step
	output = [start + x*step  for x in range(int(length))]
	return output

# Creating a array filled with zeros and the same shape
def zeros_like(inputs):
	for i in range(len(inputs)):
		for j in range(len(inputs[i])):
			inputs[i][j] = 0
	return inputs

# Eye function: turning a list in one-hot vector
def eye(length, true_values):
	example = []
	output = []
	for i in range(length):
		example.append([1 if x==i else 0 for x in range(length)])
	for i in range(len(true_values)):
		output.append(example[true_values[i]])
	return output

# Eye function but a vector as a diagonal
def diagflat(inputs):
	output = []
	for i in range(len(inputs)):
		output.append([inputs[i] if x==i else 0 for x in range(len(inputs))])
	return output

import random

def random_binomial(n, p, size): 
	example_output = [1 for x in range(size)] ## Have the same dimension than output of neuron
	while True: # repeat as long as necessary
		index = random.randint(0, len(example_output) - 1)	# Randomly choose index and set value to 0
		example_output[index] = 0		# We might set an index that already is zeroed
		dropped_out = 0					# There are different ways of overcoming this problem,
		for value in example_output:	# for simplicity we count values taht are exactly 0
			if value == 0:				# while it's extremely rare in real model,
				dropped_out  += 1		# that weights are exactly 0, this is not the best method for sure				
		if dropped_out / len(example_output) >= 1-p:	# If required number of outputs is zeroed 
			break												# - leave the loop
	return example_output
print(random_binomial(1,0.2,10))

# Gives the absolute value back
def abs(value): 
	if value < 0:
		return value  * -1
	else: 
		return value 

# Gives the sign in return
def sign(value):
	if value < 0:
		return -1
	elif value == 0:
		return 0
	else:
		return 1

# Calculates the standard deviation to the mean
def std(values):
	length = len(values)
	adder = 0
	for e in values:
		adder += e
	mean = adder / length # Calculate mean
	adder = 0 # Reset adder

	for e in values:
		adder += (e-mean)**2
	std = (adder / (length - 1))**0.5
	return std

def reshape(y, axis1, axis2):
	pass

	


