# Testing Camera to the Model
import sys
sys.path.insert(1, 'D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/NNFS')

from nnma import *
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras

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

keras_model = keras.models.load_model('D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/NNFS/Optimizing_RPS/testKeras')
model = Model.load('D:/Colin Berger/Documents/Andere Benutzer/Aloïs/MA_Aloïs/Neural_Network_Github/NNFS/Optimizing_RPS/test/Network.model')

RPS_labels = {
	0: 'rock',
	1: 'paper', 
	2: 'scissors'
}

image_datas = []

cv2.namedWindow("Preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
	rval, frame = vc.read()
else:
	rval = False

while rval:
	cv2.imshow("Preview", frame)
	rval, frame = vc.read()
	key = cv2.waitKey(20)
	data = frame
	cv2.imwrite('cam/im.png', data)
	image_data = cv2.imread('cam/im.png', cv2.IMREAD_GRAYSCALE)
	image_data = cv2.resize(image_data, (28,28))
	# image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
	# confidences = model.predict(image_data)
	# predictions = model.output_layer_activation.predictions(confidences)
	# prediction = RPS_labels[predictions[0]]
	image_data = np.expand_dims(image_data, -1)
	image_data = image_data.astype('float16')
	image_datas.append(image_data)
	image_datas.append(image_data)
	image_datass = np.array(image_datas)
	prediction = keras_model.predict(image_datass)
	print(prediction)
	if key == 27: # exit on ESC
		break



cv2.imwrite('cam/im.png', data)
image_data = cv2.imread('cam/im.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28,28))
img = image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
prediction = RPS_labels[predictions[0]]
print(prediction)
print(confidences)
plt.imshow(img, cmap='gray')
plt.show()

vc.release()
cv2.destroyWindow("Preview")
