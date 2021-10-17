# Prepare image
import cv2
import matplotlib.pyplot as plt	
import numpy as np

image_data = cv2.imread('Shirt.png', cv2.IMREAD_GRAYSCALE)

image_data = cv2.resize(image_data, (28,28))

image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

plt.imshow(image_data, cmap='gray')
plt.show()