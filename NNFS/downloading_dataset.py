# downloading_dataset

import os
import urllib
import urllib.request

# First retrieve the dataset from the nngs.io site.
URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE = 'fashion_mnist_images.zip'
FOLDER = 'fashion_mnist_images'

from zipfile import ZipFile

if not os.path.isfile(FILE):
	print(f'Downloading {URL} and saving as  {FILE}...')
	urllib.request.urlretrieve(URL, FILE)

print('Unzipping images...')
with ZipFile(FILE) as zip_images:
	zip_images.extractall(FOLDER)

print('Done!')

# It's common to go from RGB to grey and normalize size (that dataset is already grayscaled and normalized)