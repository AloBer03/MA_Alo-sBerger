# Optimazation:

# Numerical Derivaties
import matplotlib.pyplot as plt
import numpy as np

def f(x):
	return 2*x**2
x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)

p2_delta = 0.0001 
# Small enough for the error to be small but big enough for the number not to be 0
# for python not rounding it to zero  (because you can't divide by 0)

x1 = 2
x2 = x1 + p2_delta
y1 = f(x1)
y2 = f(x2)

print((x1, y1), (x2, y2))

colors = ['k', 'g', 'r', 'b', 'c']

approximate_derivative = (y2-y1)/(x2-x1)
print(approximate_derivative)
# This is called numerical differentiation

# Graph tangent
# Calculate b (y=mx+b => y-mx = b)
b= y2 - approximate_derivative*x2

# Put it into a function so we can calculate it mulpiple times (m and b are a constant)
def approximate_tangent_line(x, approximate_derivative):
	return (approximate_derivative*x) + b

for i in range(5):
	p2_delta = 0.0001
	x1 = i
	x2 = x1+p2_delta

	y1 = f(x1)
	y2 = f(x2)

	# print((x1, y1), (x2, y2))
	approximate_derivative = (y2-y1) / (x2-x1)
	b = y2 - (approximate_derivative*x2)

	# Plotting the tangent line for values +/- 0.9
	to_plot = [x1-0.9, x1, x1+0.9]
	plt.scatter(x1, y1, c=colors[i])
	plt.plot([point for point in to_plot],
			 [approximate_tangent_line(point, approximate_derivative) for point in to_plot],
			 c=colors[i])
	# print('Approximate derivative for f(x)', f'where x = {1} is {approximate_derivative}')


# plt.show()

# Backpropagation with chain rule on a single Neuron

# Forward pass
x = [1.0, -2.0, 3.0] # Inputs
w = [-3.0, -1.0, 2.0] # Weights
b = 1.0 # Bias

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding together
z = xw0 + xw1 + xw2 + b
 
# ReLU activation function
y = max(z,0)
print(y)

# Backward pass

# The derivative from the next layer
dvalue = 1.0

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z >0 else 0.)

# Partial derivatives of the multiplikation, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw0
drelu_dxw2 = drelu_dz * dsum_dxw0
drelu_db = drelu_dz * dsum_db

# One step further
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2



print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)


# All together
drelu_dx0 = dvalue * (1. if z > 0 else 1.0) * w[0] # p.202

dx = [drelu_dx0, drelu_dx1, drelu_dx2] # Gradients on inputs
dw = [drelu_dw0, drelu_dw1, drelu_dw2] # Gradients on weights
db = drelu_db # Gradient on bias...just  1 bias here

print(w, b)

# To optmize we apply a negative fraction of the gradient (to get a smaller neuron output
# When loss function added it will decrease the loss functions output) 
# This works because the derivative tells you where you go down (in which direction your looking)
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

print(w, b)

# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# Adding biases
z = xw0 + xw1 + xw2 + b
 
# ReLU activation function
y = max(z,0)

print(y)


#B ackpropagation with multiple Neurons
 
# Passed-in gradient from the next layer
# for purpose of this example we're going to use a vector of 1s
dvalues = np.array([[1.,1.,1.]])

# We have 3 sets of weights - one set for each neuron
# we have 4 inputs,thus 4 weights
# Recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
					[0.5, -0.91, 0.26, -0.5],
					[-0.26, -0.27, 0.17, 0.87]]).T
# Sum weights related to the given input multiplied by the gradient related to given Neuron
dx0 = sum([weights[0][0]*dvalues[0][0], weights[0][1]*dvalues[0][1], weights[0][2]*dvalues[0][2]])
dx1 = sum([weights[1][0]*dvalues[0][0], weights[1][1]*dvalues[0][1], weights[0][2]*dvalues[0][2]])
dx2 = sum([weights[2][0]*dvalues[0][0], weights[2][1]*dvalues[0][1], weights[2][2]*dvalues[0][2]])
dx3 = sum([weights[3][0]*dvalues[0][0], weights[3][1]*dvalues[0][1], weights[3][2]*dvalues[0][2]])
# We have one partial derivative for each neuron 
# and mulpiply it by the neuron's pertial derivative with resperct to its input
# Sum ( each inputs* its respective weights ) =^^^^
dinputs =  np.array([dx0, dx1, dx2, dx3])
# This actually equals a dot pruduct so 
dinputs = np.dot(dvalues[0], weights.T)


# When batches of gradient come in
dvalues = np.array([[1., 1., 1.],
					[2., 2., 2.],
					[3., 3., 3.]])
# Weights the same
weights = np.array([[0.2, 0.8, -0.5, 1],
					[0.5, -0.91, 0.26, -0.5],
					[-0.26, -0.27, 0.17, 0.87]]).T
dinputs = np.dot(dvalues, weights.T)

# Calculate the gradient respective to the weights
dvalues = np.array([[1., 1., 1.],
					[2., 2., 2.],
					[3., 3., 3.]])
# Now inputs instead of weights (derivateive of weights = inputs x*y => f'(x) = y)
# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
				   [2., 5., -1., 2],
				   [-1.5, 2.7, 3.3, -0.8]])
# Sum weights of given input and multiply by the passed-in gradient for this neuron
# We want the shape to match the shape of weights so we need to transpose inputs = 4,3
# First Parameter of first input = first parameter of output
# Second parameter of second input = second parameter of output

# This is the gradient of the neuron function with respect to the weights
dweights = np.dot(inputs.T, dvalues)

# Now biases
# Passed-in gradient from the next layer 
dvalues = np.array([[1., 1., 1.],
					[2., 2., 2.],
					[3., 3., 3.]])

# One bias for each neuron
# Biases are the row  vector with shape (1, neurons)
biases = np.array([[2, 3, 0.5]])

# Derivative of biases is always one so we can just sum dvalues and skip multiplying it with biases
dbiases = np.sum(dvalues, axis=0, keepdims=True)


# And finally the ReLU function
# Example layer output
z = np.array([[1, 2, -3, -4],
			  [2, -7, -1, 3],
			  [-1, 2, 5, -1]])

dvalues = np.array([[1, 2, 3, 4],
					[5, 6, 7, 8],
					[9, 10, 11, 12]])

# ReLU activation's derivative
drelu = np.zeros_like(z)
drelu[z>0] = 1.0
# Chain rule
drelu *= dvalues

# Simplified
drelu = dvalues.copy()
drelu[z <= 0] = 0

# print(dinputs)
# print(weights)
# print(dbiases)
# print(drelu)


# Forward and backwards pass through a full layer
# Passed-in values of the next layer
dvalues = np.array([[1., 1., 1.],
					[2., 2., 2.],
					[3., 3., 3.]])
# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
				   [2., 5., -1., 2],
				   [-1.5, 2.7, 3.3, -0.8]])
# 3 sets of weights for each Neuron and 4 weights per set for 4 inputs 
weights = np.array([[0.2, 0.8, -0.5, 1],
					[0.5, -0.91, 0.26, -0.5],
					[-0.26, -0.27, 0.17, 0.87]]).T
# 3 biases for 3 neurons
biases = np.array([[2, 3, 0.5]])

# Forward pas
layer_outputs = np.dot(inputs, weights) + biases 
relu_outputs = np.maximum(0, layer_outputs)

# Now optimize and bockpropagate
# First pass through ReLU function _/
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0

# Denselayer
# dinputs - ,multiply by weights
dinputs = np.dot(drelu, weights.T)
# Weights multiply by inputs
dweights =  np.dot(inputs.T, drelu)
# Biases sum
dbiases = np.sum(drelu, axis=0, keepdims=True)

# update parameters
weights += -0.01 * dweights
biases += -0.01 * dbiases

print(weights)
print(biases)

# => update the ReLU and Layer_dense class with a backwards function =>
# Add self.inputs to rember the input 

# Now also calculate the derivative of the Categorical Cross-Entropy loss function
# How? p.215 math
# Directly implement it in the class

# Lastly the derivative of the Softmax activation
# p.220
# Tested here but then directly implemented
softmax_output = [0.7, 0.1, 0.2]

softmax_output = np.array(softmax_output).reshape(-1,1)
print(softmax_output)
print(np.diagflat(softmax_output)) # First part "Sj*kronecker delta" - Sj * Sk
print(np.dot(softmax_output, softmax_output.T)) # Second part Sj*kronecker delta - "Sj * Sk"
# All together
print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))
# This is called a Jacobian matrix (here an array of partial derivatives) p.228
# Implement it

# But there is a way to comibine both which makes it way faster
# Note here the classes Activation-softamx and Categorical Crossentropy are missing

# Softmax classifier - combined Softmax activation and cross-entropy loss for faster backwards step
class Activation_Softmax_Loss_CategoricalCrossentropy():

	# Creates activation and loss function objects
	def __init__(self):
		self.activation = Activation_Softmax() 
		self.loss = Loss_CategoricalCrossentropy()

	# Froward pass
	def forward(self, inputs, y_true):
		#output layer's activation function
		self.activation.forward(input)
		#set output
		self.output = self.activation.output
		#calculate and return the loss value
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
		self.dinputs[range(samples), y_true] -= 1.0
		# Normalize gradient
		self.dinputs = self.dinputs / samples