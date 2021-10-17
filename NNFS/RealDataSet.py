# Real data set

import nnfs
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

nnfs.init()

labels = os.listdir('fashion_mnist_images/train')
print(labels)

files = os.listdir('fashion_mnist_images/train/0')
print(files[:10])
print(len(files))

# We have 6000 samples per class 
# This ensures a balanced distribution over the diffrent classes
# else they become bias because they seek for the steepest gradient mostly ending in a local minimum

# p.537

image_data = cv2.imread('fashion_mnist_images/train/7/0002.png', cv2.IMREAD_UNCHANGED)

np.set_printoptions(linewidth=200)

print(image_data)

plt.imshow(image_data, cmap= 'gray')
plt.show()

# Scan all the directories and create a list of labels

# Create lists for samples and labels
#X = []
#y = []

# For each label folder
# for label in labels:
#	 # And for each image in given folder
#	 for file in os.listdir(os.path.join('fashion_mnist_images/train', label, file)):
#	 	# os.path.join creates a path by joining the following parameters with a '/'
#	 	# Here resulting in 'fashion_mnist_images/train/"label"/"file"'
#	 	# Read the image
#	 	image = cv2.imread(os.path.join('fashion_mnist_images/train', label, file), cv2.IMREAD_UNCHANGED)
# 
#	 	# And append it and a label to the lists
#	 	X.append(image)
#	 	y.append(label)

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

	# Scan all the directories and create a list of labels
	labels = os.listdir(os.path.join(path, dataset))

	# Create lists for the samples and labels
	X = []
	y = []

	# For each label folder 
	for label in labels:
		# And for each image in given folder
		for file in os.listdir(os.path.join(path, dataset, label)):
			# Read the image 
			image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

			# And append it and a label to the lists
			X.append(image)
			y.append(label)

	# Convert the data to proper numpy arrays and return 
	return np.array(X), np.array(y).astype('uint8') # say that y is int and not float

# MNIST dataset (train + test)
def create_data_mnist(path):

	# Load both sets seperately
	X, y = load_mnist_dataset('train', path)
	X_test, y_test = load_mnist_dataset('test', path)

	# And return all the data
	return X, y, X_test, y_test

# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')


# Data preprocessing

# Now we have to scale the data, because Neural Network work best with numbers between the range 
# 0 and 1 or -1 and 1
# Make sure to scale train and testing data


# Here we will use a linear scaling method. For any scaling method expect linear one you should never scale
# it according to the test dataset (for instance if you need a maximum and a minimum don't include the testing data)

# Scale features
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

# print(X.min(), X.max())

# Next up we have to chnage the shape of the data because the model needs a 1 dimensional vector
# but the images are stored in 28 by 28. 
# reshape(6000, -1) will keep the 6000 samples but make the two other as one dimension

# Reshape to vectors
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# You can also reshaping it "yourself" by writing X.reshape(X.shape[0], X.shape[1]*X.shape[2])


# Data shuffling

# Now we need to shuffle the data else the neural network will just predict the class he is passing 
# Through without really learning to identify the object

# We need to be careful not to mix the the sample and the target the same else it won't be right
# To do this we'll create a key the same size of the training data and shuffle the key instead
# afterwrads we will shuffle both the sample and the target using that same key

keys = np.array(range(X.shape[0]))
print(keys[:10])

np.random.shuffle(keys)
print(keys[:10])

X = X[keys]
y = y[keys]

print(y[:15])


# Batches

# To this point we trained the entire data at once. But for real life dataset who are much bigger
# it's preferable to train in batches. Common batch sizes range between 32 and 128 samples.
# You may want to go smaller if you have trouble fitting everyting into memory (which would
# eventually make it slower), or larger if you want training to go faster. Though you should have
# a big enough batch so that it is somewhat representative of your data. If you make the batches 
# to big it will start you'll see accuracy drop down and loss go up.

# Each batch of samples being trained is reffered to as a step
# steps = X.shape[0] // BATCH_SIZE # No fraction step
# if steps * BATCH_SIZE < X.shape[0]:
# 	steps += 1
# for epoch in range(EPOCHS):
# 	for step in range(steps):
#		batch_x = X[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
#		batch_y = y[step*BATCH_SIZE:(step+1)*BATCH_SIZE]

# To display the loss and accuracy intelligently we have to modify how we present loss and accuracy
# Now we want to know batch-wise statistics and epoch-wise. For over all accuracy and loss, we want
# to calculate a sample-wise average. To do this, we will accumulate the sum of losses from all
# batches and counts to calculate the mean value at the end of each epoch.
# We'll change that in the coommon Loss class' calculate method:

# We will do the same thing with the Accuracy class

# Now let's change the train function 
# First we add a new parameter called batch_size
# Then we calculate how many steps ther will be (default 1)
# Next, we'll modify the loop:
# - Print the epoch number
# - Reset loss and accuracy
# - Iterate over steps
# - Change the way the Summary is printed

# We need to add batching to the validation data, because it will certainly be bigger then batch size

# Now we are ready to train using batches

# We are trying without shuffling:
# We quickly see that the model learns quite quickly by predicting the same class over and over again
# Although it sometimes reaches an accuracy of 100% at the end the overall accuracy is quite low 0.2

