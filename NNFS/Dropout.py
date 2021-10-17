#Dropout 

# You drop out random neurons so they don't function anymore
# This makes sure that the Neural system doesn't relie on specific neurons but can work with all of them

import random
import numpy as np


dropout_rate = 0.5
# Example output containing 10 values
example_output = [0.27, -1.03, 0.67, 0.99, 0.05,
				  -0.37, -2.01, 1.13, -0.07, 0.73]

while True: # repeat as long as necessary

	# Randomly choose index and set value to 0
	index = random.randint(0, len(example_output) - 1)
	example_output[index] = 0

	# We might set an index that already is zeroed
	# There are different ways of overcoming this problem,
	# for simplicity we count values taht are exactly 0
	# while it's extremely rare in real model that weights are exactly 0, this is not the best method for sure
	dropped_out = 0
	for value in example_output:
		if value == 0:
			dropped_out  += 1

	# If required number of outputs is zeroed - leave the loop
	if dropped_out / len(example_output) >= dropout_rate:
		break

print(example_output)
# note that the dropout rate is about the neurons we DISABLED (q) (TensorFlow / Keras)
# sometimes we mean the one's we KEEP (p) (PyTorch)

# We use it for training, which means that the prediction will be 1/q times lower which can mess up the system

# We could just divide the prediction by q but it would be an extra step 
# there for scale the data back up after dropout during training to mimic all neurons beeing active
example_output	= np.array(example_output)
example_output *= np.random.binomial(1, 1-dropout_rate,  example_output.shape) / (1-dropout_rate)
print(example_output)

# Surely it won't be the same but very similar

dropout_rate = 0.25

print(f'sum initial {sum(example_output)}') # Sum before dropout

sums = []
for i in range(10000):
	# Do dropout and scale it back up with 1-dropout_rate
	example_output2 = example_output * np.random.binomial(1, 1-dropout_rate, example_output.shape) / \
					  (1-dropout_rate)
	sums.append(sum(example_output2)) # Collect all sums to make an average

print(f'mean sum: {np.mean(sums)}')

# Quite similar if oy do it a thousand times, therefor no need to worry about the prediction beeing scaled badly


# Backwards pass  f(z,q) = z/(1-q) => 1/(1-q) if "r" is 1 and if it is 0 the derivative is also 0

# Implement it in full code as a new layer 
