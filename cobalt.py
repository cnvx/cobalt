#!/usr/bin/env python3

import matplotlib.pyplot as plt
import tensorflow as tf
import prettytensor as pt
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import sys
import urllib.request
import tarfile
import zipfile
import pickle

''' Get the CIFAR-10 data set '''

# Where to save the data set
data_path ='./CIFAR-10/'

# Where to get the data set from
data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# Image dimensions
image_size = 32
image_size_cropped = 24

number_of_channels = 3
number_of_classes = 10

number_of_files_train = 5
images_per_file = 10000

# Total number of images in the training set
number_of_images_train = number_of_files_train * images_per_file

# Length of an image when flattened to a 1-dim array.
image_size_flat = image_size * image_size * number_of_channels

# Get full file path, return directory if called with no filename
def get_file_path(filename = ''):
    return os.path.join(data_path, 'cifar-10-batches-py/', filename)
    
# Unpickle file and return data
def unpickle(filename):
    file_path = get_file_path(filename)
    print('Loading data from ' + file_path)

    with open(file_path, mode = 'rb') as file:
        data = pickle.load(file, encoding = 'bytes')

    return data

# Convert from CIFAR-10 format to 4 dimensional array
def convert_images(unconverted):
    float_array = np.array(unconverted, dtype = float) / 255.0

    # Reshape to 4 dimensions
    images = float_array.reshape([-1, number_of_channels, image_size, image_size])
    images = images.transpose([0, 2, 3, 1])

    return images

# Load pickled data and return converted images and their class numbers
def load_data(filename):
    data = unpickle(filename)

    # Get the unconverted images, 'data' is a bytes literal
    unconverted = data[b'data']

    # 'labels' are byte literals
    class_numbers = np.array(data[b'labels'])

    images = convert_images(unconverted)

    return images, class_numbers

# Download and extract the data set if it doesn't already exist
def download_data_set():

    # Output the download progress
    def print_progress(count, block_size, total_size):
        percentage = float(count * block_size) / total_size
        message = '\rDownloading... {0:.1%}'.format(percentage)

        sys.stdout.write(message)
        sys.stdout.flush()

    # Add the filename from the URL to the download_directory
    filename = data_url.split('/')[-1]
    file_path = os.path.join(data_path, filename)

    # Check if the file already exists
    if not os.path.exists(file_path):
        
        # Check if the download directory exists
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        
        # Download
        file_path, _ = urllib.request.urlretrieve(url = data_url,
                                                  filename = file_path,
                                                  reporthook = print_progress)

        print('\rDownload complete\nExtracting files...')
        
        # Extract
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_path)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_path)

        print('Done')

    else:
        print('Data has already been downloaded and extracted')

# Return a list of class names
def load_class_names():
    unconverted = unpickle(filename = 'batches.meta')[b'label_names']

    # Convert class names to strings
    converted = [x.decode('utf-8') for x in unconverted]

    return converted

# Return a one hot encoded (helps reduce errors) 2 dimensional array of class numbers and labels
def one_hot_encoded(class_numbers, number_of_classes = None):
    if number_of_classes is None:
        number_of_classes = np.max(class_numbers + 1)
        
    return np.eye(number_of_classes, dtype = float)[class_numbers]
    
# Load all the training data
def load_training_data():

    # Preallocate arrays, helps with efficiency
    images = np.zeros(shape=[number_of_images_train, image_size, image_size, number_of_channels], dtype = float)
    class_numbers = np.zeros(shape=[number_of_images_train], dtype = int)

    start_at = 0

    # Load each data file
    for i in range(number_of_files_train):
        images_batch, class_numbers_batch = load_data(filename = 'data_batch_' + str(i + 1))
        number_of_images_in_batch = len(images_batch)

        end = start_at + number_of_images_in_batch
        
        # Populate arrays
        images[start_at:end, :] = images_batch
        class_numbers[start_at:end] = class_numbers_batch
                
        # Start where the last run ended
        start_at = end

    return images, class_numbers, one_hot_encoded(class_numbers = class_numbers, number_of_classes = number_of_classes)

# Load all the test data
def load_test_data():
    images, class_numbers = load_data(filename='test_batch')
    return images, class_numbers, one_hot_encoded(class_numbers = class_numbers, number_of_classes = number_of_classes)

download_data_set()
images_train, classes_train, labels_train = load_training_data()
images_test, classes_test, labels_test = load_test_data()

''' Placeholder varialbes for the neural network '''

# Images used as input
x = tf.placeholder(tf.float32, shape = [None, image_size, image_size, number_of_channels], name = 'x')

# Real lables associated with each image
y_actual = tf.placeholder(tf.float32, shape = [None, number_of_classes], name = 'y_actual')

# Real class numbers
y_actual_class_numbers = tf.argmax(y_actual, axis = 1)
