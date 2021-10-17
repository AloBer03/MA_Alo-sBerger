# Test

import PIL
import numpy as np
import pandas
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import cv2
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from scipy import misc, ndimage
import imageio



num_classes = 10
input_shape = (28, 28, 1)

gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=25,width_shift_range=0.10, height_shift_range=0.10,
                         shear_range=0.25, zoom_range=0.1, channel_shift_range=10., horizontal_flip=True)

# Loads a MNIST dataset
def store_from_to_mnist_dataset(dataset, path, store, storing_path, resize, shape):

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
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            # if resize, resize the image according to shape
            if resize:
                image = cv2.resize(image, shape)

            # If store, store the image
            if store:
                status = cv2.imwrite(os.path.join(storing_path, dataset, label, 'img-'+str(label)+'-'+str(i)+'.jpg'), image)
            
            # And append it and a label to the lists and save the new image
            X.append(image)
            y.append(label)
            i += 1
            print(status)

    # Convert the data to proper numpy arrays and return 
    return np.array(X), np.array(y).astype('uint8')

# Store a dataset from a list of images
def store_from_list_to_dataset(dataset, path, labels):

	statuses = []
	i = 0
	for image, label in zip(dataset, labels):

		status = cv2.imwrite(os.path.join(path, str(label), 'img-'+str(label)+'-'+str(i)+'.jpg'), image)
		print(status)
		statuses.append(status)
		i+=1

	return statuses


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
            image = imageio.imread(os.path.join(path, dataset, label, file))

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

# Augment a image
def aug_img(image, n):

	aug = gen.flow(image)
	aug_images = [next(aug)[0].astype(np.uint8) for i in range(n)]

	return aug_images



#The data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, y_train, x_test, y_test = create_data_mnist('D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/NNFS/RockPaperScissors/Data')

dataset = []

pathname= "rock.png"
image_data = cv2.imread('rock.png', cv2.IMREAD_UNCHANGED)
image_data = cv2.resize(image_data, (28,28))
image_data = 255 - image_data
image_data = np.expand_dims(image_data, 0)
print(image_data)
image = np.expand_dims(imageio.imread(pathname),0)
image_d = imageio.imread(pathname)
# Print diffrent Images to see the diffrent format their in
# print('image_data')
# print(image_data)
# print(image_data.shape)
# print('x_train')
# print(x_train)
# print(x_train.shape)
# print('image')
# print(image)
# print(image.shape)
# print('end')

path = 'D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/RockPaperScissors/Augmented_Data/train'
factor = 0.5
# x_training = cv2.resize(x_train[0],(28,28))
for img in x_train:
	imag = np.expand_dims(img,0)
	augmented = aug_img(imag, factor)
	for im in augmented:
		dataset.append(im)
print('done')
print(y_train)
print(len(y_train))
y_train_augmented = []
for i in y_train:
	for j in range(factor):
		y_train_augmented.append(i)
print(len(y_train_augmented))
stat = store_from_list_to_dataset(dataset, path, y_train_augmented)
print(stat)

#picture = PIL.Image.open(pathname)
#picture = picture.save('ro.jpg')

#cv2.imwrite('roc.jpg', image[0])
#p = 'D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network'

aug_iter = gen.flow(image)
#aug_iter = gen.flow(np.expand_dims(train[0],0))
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(1)]
# os.mkdir('test')
# cv2.imwrite('test/rocks.jpg', image)
#print(cv2.imwrite('test/test/rocks.jpg', aug_images[0]))


# Show 10 Augmented images
fig, axs = plt.subplots(nrows=2, ncols=5)
i = [x for x in range(1)]
for ax, i in zip(axs.flat, i):
    ax.imshow(aug_images[i])
plt.show()

# Do it in each seperate folder...


