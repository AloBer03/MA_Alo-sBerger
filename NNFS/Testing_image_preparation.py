# Testing image preperation
import cv2
import matplotlib.pyplot as plt	
import numpy as np
import os

image_data = np.array([cv2.imread('Shirt.png', cv2.IMREAD_GRAYSCALE),cv2.imread('Shirt.png', cv2.IMREAD_GRAYSCALE)])
image_data_test = np.array([cv2.imread('Shirt.png', cv2.IMREAD_GRAYSCALE),cv2.imread('Shirt.png', cv2.IMREAD_GRAYSCALE)])

for img, img_test in zip(image_data, image_data_test):
	print("hi")
	img = cv2.resize(img, (28,28))
	img_test = cv2.resize(img, (28,28))
print(image_data)


image_data = (image_data.reshape(image_data.shape[0], -1).astype(np.float32) - 127.5) / 127.5
print(image_data)

plt.imshow(image_data, cmap='gray')
plt.show()

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path, store, storing_path, resize, shape):

	# Scan all the directories and create a list of labels
	labels = os.listdir(os.path.join(path, dataset))

	# Create lists for the samples and labels
	X = []
	y = []
	i = 0
	# For each label folder 
	for label in labels:
		# And for each image in given folder
		for file in os.listdir(os.path.join(path, dataset, label)):
			# Read the image 
			image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_GRAYSCALE)
			print(image)

			# if resize, resize the image according to shape
			if resize:
				image = cv2.resize(image, shape)

			# If store, store the image
			if store:
				status = cv2.imwrite(os.path.join(storing_path, dataset, label, 'img-'+str(label)+'-'+str(i)+'.png'), image)
			
			# And append it and a label to the lists and save the new image
			X.append(image)
			y.append(label)
			i += 1
			print(status)

	# Convert the data to proper numpy arrays and return 
	return np.array(X), np.array(y).astype('uint8') # say that y is int and not float

# MNIST dataset (train + test)
def create_data_mnist(path, store=False, storing_path=None, resize=False, shape=[28,28]):

	# Load both sets seperately
	X, y = load_mnist_dataset('train', path, store, storing_path, resize, shape)
	X_test, y_test = load_mnist_dataset('test', path, store, storing_path, resize, shape)

	# And return all the data
	return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist('RockPaperScissors/Data',
								   True, 'RockPaperScissors/Data_resized28', True, [28,28])

