# Testing

import random
import numpy as np
from matplotlib import pyplot as plt, patches
from nnfs.datasets import spiral_data

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

# print(example_output)

length= 1
width = 4
plt.rcParams["figure.figsize"] = [7, 5]
plt.rcParams["figure.autolayout"] = True
figure, ax = plt.subplots(1)
rectangle = [ x for x in range(7)]
for i in range(len(rectangle)):
	if i%2 ==0:
		rectangle[i] = patches.Rectangle((i, 1), 1, 1, edgecolor='orange', facecolor="red", linewidth=1)
	else:
		rectangle[i] = patches.Rectangle((i, 1), 1, 1, edgecolor='orange', facecolor="green", linewidth=0)
		
# rectangle[1] = patches.Rectangle((4, 4), 1, 1, edgecolor='orange',
# facecolor="green", linewidth=7)
# rectangle[0] = patches.Rectangle((1, 2), 1, 1, edgecolor='green', facecolor='orange', linewidth= 5)
ax.plot([2, 4, 5, 1, 4, 8, 0], c='red')
for point in rectangle:
	ax.add_patch(point)



# print(np.log(0))

a = np.array([True, False, True])
# print(a)

b = a*1
# print(b)

# plt.show()

X, y = spiral_data(samples=100, classes=5)
# print(X, y)

for point, classes in zip(X,y):
	pass
	if classes==0:
		plt.scatter(point[0],point[1], color='red')
	if classes==1:
		plt.scatter(point[0],point[1], color='blue')
	if classes==2:
		plt.scatter(point[0],point[1], color='green')
	if classes==3:
		plt.scatter(point[0],point[1], color='yellow')
	if classes==4:
		plt.scatter(point[0],point[1], color='purple')
# plt.show()

for x,y in zip([1,2,3,1,2,3,1,2,3],[1,1,1,2,2,2,3,3,3]):
	plt.scatter(x,y,color = (x/3, y/3, 0))

# plt.show()

example = np.array([[1, 2], [3, 4]])
flattened = example.reshape(-1)

# print(example)
# print(example.shape)

# print(flattened)
# print(flattened.shape)
weights = np.random.randn(2, 4)
inputs = np.random.randn(6,2)


output = [[0 for x in range(len(weights[1]))] for x in range(len(inputs))]
for i_batch in range(len(output)): # Batch

	for j_neuron in range(len(output[i_batch])): # Neuron
		count = 0
		for k_input in range(len(weights)):
			count += weights[k_input][j_neuron]*inputs[i_batch][k_input]
		output[i_batch][j_neuron] = count
print(output)





