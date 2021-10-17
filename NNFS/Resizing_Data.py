# Resizing Data and store

import numpy as np
import os
import cv2

# Store mnist dataset
def store_mnist_dataset(dataset, path, store, storing_path, resize, shape):

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
    return np.array(X), np.array(y).astype('uint8')

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

# Resize
def resize(dataset, shape):
	datas = []
	for data in dataset:
		datas.append(np.resize(data, shape))
	return np.array(datas)


store_mnist_dataset('train','RockPaperScissors/Augmented_Data', store=True,
					storing_path='RockPaperScissors/Augmented_Data_resized28', resize=True, shape=(28,28))
store_mnist_dataset('test','RockPaperScissors/Data', store=True,
					storing_path='RockPaperScissors/Augmented_Data_resized28', resize=True, shape=(28,28))
