# Optimizer_functions

## code from NNFS
## My own comments are marked with ##
## My own code start with ##-- and ends with --##

# Stochastic Gradient Descent (SGD)
# SGD optimizer
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
			layer.weight_momentums = weights_updates

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


starting_learning_rate = 1.0
learning_rate_decay = 0.1
step = 1

learning_rate = starting_learning_rate * (1 / (1+learning_rate_decay*step))
print(learning_rate)

starting_learning_rate = 1.
learning_rate_decay = 0.1
step = 20

learning_rate = starting_learning_rate * (1 / (1+learning_rate_decay*step))
print(learning_rate)

starting_learning_rate = 1.
learning_rate_decay = 0.1

for step in range(20):
	learning_rate = starting_learning_rate * (1. / (1 + learning_rate_decay * step))
	print(learning_rate)
## Update the optimizer function 

## Use momentum to not get stuck in local minimas
## => update the class

## AdaGrad optimizer
## The learning rate of the wegihts with big values decrease faster compared to those 
## with smaller values. This is done to involve all neuron and not just have some with to high values and 
## other having pretty much zero impact
## Only used in specific cases because it can stall out (division by a number that grows)

## It is smiliar to vanilla SGD but normalization takes place

# AdaGrad optimizer
class Optimizer_Adagrad:

	# Initialize optimizer - set settings,
	# learning rate of 1. is default for this optimizer
	def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		

	# Call once before any parameter updates 
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay *self.iterations))

	#update parameters 
	def update_params(self, layer):
		# Adagrad: cache

		# If layer does not contain cache arrays, create them filled with zeros
		if not hasattr(layer, 'weight_chache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			# The array doesn't exist for biases yet either
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

## RMSProp
## An other adaptation to SGD
## Root Mean Square Propagation. SImiliar to AdaGrad but calculates the adaptive learning per parameter diffrently

## cache = rho * cache + (1-rho) * gradient **2
## It is similar to both momentum and Adagrad (first part similiar to momentum while second recembles Adagrad)
## Adagrad => cache = gradient **2
## momentum => update = + (momentum * previous_change) - (current_learning_rate * dweights)
## Its like constanly updating cache with a fraction of itself + a fraction of the "new cache"
## Benefits: it doesn't stall like Adagrad did (cache never gets to big for division (to big division = no impact))

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

## Adam
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