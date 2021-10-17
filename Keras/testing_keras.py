# Test

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



num_classes = 3
input_shape = (28, 28, 1)

# Loads a MNIST dataset
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

# The data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, y_train, x_test, y_test = create_data_mnist('RockPaperScissors/Augmented_Data_resized28')

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# # Scale images to the [0, 1] range
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
# # Make sure images have shape (28, 28, 1)
# x_train = np.expand_dims(x_train, -1)
# x_test = np.expand_dims(x_test, -1)
# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")

def default_datagen():
    datagen = ImageDataGenerator( fill_mode='constant', dtype=int)
    datagen.fit(data)
    return datagen

def plot_augmentation(datagen, data, n_rows=1, n_cols=5):
    n_images = n_rows * n_cols
    gen_flow = datagen.flow(data)

    plt.figure(figsize=(n_cols*4, n_rows*3))
    for image_index in range(n_images):
        image = next(gen_flow)
        plt.subplot(n_rows, n_cols, image_index+1)
        plt.axis('off')
        plt.imshow(image[0], vmin=0, vmax=255)



# pathname= "rock.png"
# image_data = cv2.imread('rock.png', cv2.IMREAD_UNCHANGED)
# image_data = cv2.resize(image_data, (28,28))
# image_data = 255 - image_data
# image_data = np.expand_dims(image_data, 0)
# print(image_data)
# image = np.expand_dims(cv2.imread(pathname, cv2.IMREAD_UNCHANGED),0)
# print('x_train')
# print(x_train)
# print('image')
# print(image)
# print('end')
# print(image.shape)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        #layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 10

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save('Optimizing_RPS/testKeras')


RPS_labels = {
    0: 'rock',
    1: 'paper', 
    2: 'scissors'
}

image_datas = []
for i in range(3):
    image_data = cv2.imread(str(RPS_labels[i])+'.png', cv2.IMREAD_GRAYSCALE)
    image_data = cv2.resize(image_data, (28,28))
    image_data = np.expand_dims(image_data, -1)
    plt.imshow(image_data, cmap='gray')
    plt.show()
    image_data = image_data.astype('float16')
    image_datas.append(image_data)

    # confidences = model.predict(image_data)
    # predictions = model.output_layer_activation.predictions(confidences)
    # prediction = RPS_labels[predictions[0]]
image_datas = np.array(image_datas)
prediction = model.predict(image_datas)
print(prediction)

