#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import sys
import urllib.request
import tarfile
import zipfile
import pickle
import glob

''' Functions for getting the CIFAR-10 data set '''

# Where to save the data set
data_path ='./CIFAR-10/'

# Where to get the data set from
data_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# Image dimensions
image_size = 32
image_size_cropped = 24

# Colour channels
number_of_channels = 3

# Possible objects
number_of_classes = 10

number_of_files_train = 5
images_per_file = 10000

# Total number of images in the training set
number_of_images_train = number_of_files_train * images_per_file

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

        sys.stdout.write('\rDownload complete    \nExtracting files...')
        sys.stdout.flush()
        
        # Extract
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_path)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_path)

        print('\rFiles extracted    ')

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

''' Image processing functions '''

# If the image is training data make random modifications, if not just crop it
def process_single_image(image, is_training_data):
    if is_training_data:
        image = tf.random_crop(image, size = [image_size_cropped, image_size_cropped, number_of_channels])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, max_delta = 0.05)
        image = tf.image.random_contrast(image, lower = 0.3, upper = 1.0)
        image = tf.image.random_saturation(image, lower = 0.0, upper = 2.0)
        image = tf.image.random_brightness(image, max_delta = 2.0)

        # Stop tf.image.random_contrast() from outputting extreme values
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image, target_height = image_size_cropped, target_width = image_size_cropped)

    return image

def process_images(images, is_training_data):
    images = tf.map_fn(lambda image: process_single_image(image, is_training_data), images)
    return images

# Get random batch

batch_size = 512

def random_batch():
    random = np.random.choice(number_of_images_train, size = batch_size, replace = False)
    
    # Select random images and labels
    x_batch = images_train[random, :, :, :]
    y_batch = labels_train[random, :]
    
    return x_batch, y_batch

''' Functions for creating weights and biases '''

def weight_variable(shape, name):
    W = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(W, name)

def bias_variable(shape, name):
    b = tf.constant(0.1, shape = shape)
    return tf.Variable(b, name)

''' Convolution and max pooling functions '''

def convolve(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

''' Variables and placeholders for the neural network '''

# Images used as input
x = tf.placeholder(tf.float32, shape = [None, image_size_cropped, image_size_cropped, number_of_channels], name = 'x')

# Real lables associated with each image
y_actual = tf.placeholder(tf.float32, shape = [None, number_of_classes], name = 'y_actual')

# Probability of not dropping neuron outputs
keep = tf.placeholder(tf.float32)

''' Neural network layers '''

# First convolutional layer
    
with tf.name_scope('first_convolutional_layer'):
    W_conv1 = weight_variable([5, 5, 3, 64], 'W_conv1')
    b_conv1 = bias_variable([64], 'b_conv1')

    # Convolve the input with the weights and add the biases
    conv1 = convolve(x, W_conv1) + b_conv1

    # Apply max pooling, reducing the image size to 12x12 pixels
    pool1 = max_pool(conv1)

    # Apply the ReLU neuron function
    relu1 = tf.nn.relu(pool1)

# Second convolutional layer

with tf.name_scope('second_convolutional_layer'):
    W_conv2 = weight_variable([5, 5, 64, 64], 'W_conv2')
    b_conv2 = bias_variable([64], 'b_conv2')

    conv2 = convolve(relu1, W_conv2) + b_conv2
    pool2 = max_pool(conv2)
    relu2 = tf.nn.relu(pool2)

# First fully connected layer

with tf.name_scope('first_fully_connected_layer'):
    # Flatten the tensor into 1 dimension
    pool2_flat = tf.reshape(relu2, [-1, 6 * 6 * 64])

    # Prepare the network variables
    W_conn1 = weight_variable([6 * 6 * 64, 256], 'W_conn1')
    b_conn1 = bias_variable([256], 'b_conn1')

    conn1 = tf.nn.relu(tf.matmul(pool2_flat, W_conn1) + b_conn1)

# Second fully connected layer

with tf.name_scope('second_fully_connected_layer'):
    W_conn2 = weight_variable([256, 128], 'W_conn2')
    b_conn2 = bias_variable([128], 'b_conn2')

    conn2 = tf.nn.relu(tf.matmul(conn1, W_conn2) + b_conn2)

# Dropout layer

with tf.name_scope('dropout_layer'):
    drop = tf.nn.dropout(conn2, keep)
    
# Output layer

with tf.name_scope('output_layer'):
    W_output = weight_variable([128, 10], 'W_output')
    b_output = bias_variable([10], 'b_output')

    # Regression
    output = tf.matmul(drop, W_output) + b_output

''' Additional functions and ops '''

# Cost function
with tf.name_scope('cost_function'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_actual, logits = output))
    tf.summary.scalar('cost_function', cross_entropy)

# Train step
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Accuracy
with tf.name_scope('network_accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_actual, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('network_accuracy', accuracy)

# Ask how many times to train
def get_times_to_train():
    try:
        times = int(sys.argv[2])
    except IndexError:
        sys.stdout.write('Input times to train: ')
        sys.stdout.flush()
        times = int(input())
        
    return times

# Initialization op
init_op = tf.global_variables_initializer()

# Save and restore op
saver = tf.train.Saver()

# Save locations
save_location = './data/cobalt.ckpt'
log_dir = './log'

# Download the data set
download_data_set()

# Check for existing network
if glob.glob(save_location + '*'):
    try:
        if str(sys.argv[1]) == 'y':
            times_to_train = get_times_to_train()
        else:
            times_to_train = 0;
    except IndexError:
        sys.stdout.write('Overwrite existing network? (y/n): ')
        sys.stdout.flush()
        if str(input()) == 'y':
            times_to_train = get_times_to_train()
        else:
            times_to_train = 0;
else:
    times_to_train = get_times_to_train()

''' Prepare the data '''

# Load the data
images_train_raw, classes_train, labels_train = load_training_data()
images_test_raw, classes_test, labels_test = load_test_data()

# Process the images
with tf.Session() as proc_sess:
    if times_to_train != 0:
        with tf.name_scope('training_image_processing'):
            images_train = process_images(images_train_raw, True).eval()
    with tf.name_scope('validation_image_processing'):
        images_test = process_images(images_test_raw, False).eval()
        
''' Train the network '''
    
# Start the session
with tf.Session() as sess:
    sess.run(init_op)

    if times_to_train != 0:
        # Merge all summaries and write them to disk
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        
        # Run the training loop, show progress every 100th step
        for i in range(times_to_train):
            x_batch, y_actual_batch = random_batch()
            feed_dict_train = {x: x_batch, y_actual: y_actual_batch, keep: 0.5}
            feed_dict_accuracy = {x: x_batch, y_actual: y_actual_batch, keep: 1}
            
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict_accuracy)
                print('Training network (step %g/%g), current batch accuracy: %g' % (i, times_to_train, train_accuracy))
                
            # Execute a training step
            summary, _ = sess.run([merged, train_step], feed_dict_train)
                
            # Write a summary
            train_writer.add_summary(summary, i)

        # Save the network variables to disk
        save_path = saver.save(sess, save_location)
        print('Network saved to %s' % save_path)
                               
    else:
        # Load the old network variables
        saver.restore(sess, save_location)
        print('Network loaded from %s' % save_location)
            
    # Get final accuracy
    feed_dict_test = {x: images_test, y_actual: labels_test, keep: 1}
    sys.stdout.write('Getting accuracy...')
    sys.stdout.flush()
    print('\rNetwork accuracy: %g' % accuracy.eval(feed_dict_test))
