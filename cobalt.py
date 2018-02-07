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
import argparse as arg

''' Hyperparameters '''

batch_size = 64
initial_weight_decay = 5e2
initial_learning_rate = 0.1
learning_rate_decay = 0.96
learning_decay_frequency = 10000

# Enable data augmentation during training
augment_data = False

''' Argument parsing '''

format = lambda prog: arg.HelpFormatter(prog, max_help_position=79)
parser = arg.ArgumentParser(formatter_class = format)

parser.add_argument('-t', '--train', dest = 'steps', type = int, default = 0,
                    help = 'enter number of times to train')
parser.add_argument('-a', '--accuracy', action = 'store_true', default = False,
                    help = 'check network validation accuracy')
parser.add_argument('-d', '--data_augmentation', action = 'store_true', default = False,
                    help = 'enable data augmentation during training')
parser.add_argument('-o', '--overwrite', action = 'store_true', default = False,
                    help = 'overwrite saved network data')
parser.add_argument('-s', '--save', metavar = 'directory', dest = 'save_dir', default = 'data',
                    help = 'network save location')
parser.add_argument('-l', '--log', metavar = 'directory', dest = 'log_dir', default = 'log',
                    help = 'where to save the log files')

args = parser.parse_args()

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
number_of_files_test = 1
images_per_file = 10000

# Total number of images
number_of_images_train = number_of_files_train * images_per_file
number_of_images_test = number_of_files_test * images_per_file

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

# Load all the validation data
def load_test_data():
    images, class_numbers = load_data(filename='test_batch')
    return images, class_numbers, one_hot_encoded(class_numbers = class_numbers, number_of_classes = number_of_classes)

''' Image processing functions '''

# If data augmentation is enabled make random modifications, if not just crop it
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
        image = tf.image.resize_image_with_crop_or_pad(image, target_height = image_size_cropped,
                                                       target_width = image_size_cropped)

    return image

def process_images(images, is_training_data):
    images = tf.map_fn(lambda image: process_single_image(image, is_training_data), images)
    return images

# Get random batch of images and labels
def random_batch(validation = False):
    if validation:
        random = np.random.choice(number_of_images_test, size = batch_size, replace = False)
        x_batch = images_test[random, :, :, :]
        y_batch = labels_test[random, :]
    else:
        random = np.random.choice(number_of_images_train, size = batch_size, replace = False)
        x_batch = images_train[random, :, :, :]
        y_batch = labels_train[random, :]
    
    return x_batch, y_batch

''' Variables and placeholders for the neural network '''

# Images used as input
x = tf.placeholder(tf.float32, shape = [None, image_size_cropped, image_size_cropped, number_of_channels], name = 'x')

# Real lables associated with each image
y_actual = tf.placeholder(tf.float32, shape = [None, number_of_classes], name = 'y_actual')

# Probability of not dropping neuron outputs
keep = tf.placeholder(tf.float32)

# Is the network training or not, used for batch normalization
is_training = tf.Variable(initial_value = False, trainable = False, name = 'is_training')

# Learning rate
with tf.name_scope('learning_rate'):
    learning_rate = tf.placeholder(tf.float32, shape=[])
    tf.summary.scalar('learning_rate', learning_rate)

''' Functions used in the neural network '''

# Create weight variable using Xavier initialization
def weight_variable(shape, name, decay):
    xavier = tf.contrib.layers.xavier_initializer(uniform = False)
    variable = tf.get_variable(name, shape = shape, initializer = xavier)

    # Weight decay
    if decay:
        weight_decay = initial_weight_decay
        for i in shape:
            weight_decay = weight_decay / i
            weight_loss = tf.multiply(tf.nn.l2_loss(variable), weight_decay, name = 'weight_loss')

    return variable

# Create bias variable
def bias_variable(shape, name):
    return tf.get_variable(name, shape = shape, initializer = tf.constant_initializer(0.0))

# Convolution
def convolve(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

# Max pooling
def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# Batch normalization
def batch_norm(x, depth, is_training):
    mean_batch, variance_batch = tf.nn.moments(x, [0, 1, 2], name = 'batch_mean_variance_calculation')
    moving_average = tf.train.ExponentialMovingAverage(decay = 0.5)
    
    def mean_variance_update():
        moving_average_op = moving_average.apply([mean_batch, variance_batch])
        with tf.control_dependencies([moving_average_op]):
            return tf.identity(mean_batch), tf.identity(variance_batch)
        
    mean, variance = tf.cond(is_training, mean_variance_update, lambda: (moving_average.average(mean_batch),
                                                                         moving_average.average(variance_batch)))
    
    # Also known as offset and scale
    beta = tf.Variable(tf.constant(0.0, shape = [depth]), name = 'beta', trainable = True)
    gamma = tf.Variable(tf.constant(1.0, shape = [depth]), name='gamma', trainable = True)
    
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-4)

''' Neural network architecture '''

# First convolutional layer

with tf.name_scope('first_convolutional_layer'):
    W_conv1 = weight_variable([5, 5, 3, 64], 'W_conv1', False)
    b_conv1 = bias_variable([64], 'b_conv1')
    
    # Convolve the input with the weights and add the biases
    conv1 = convolve(x, W_conv1) + b_conv1
    
    # Apply the rectified linear unit function
    relu1 = tf.nn.relu(conv1)
    
    # Apply max pooling, reducing the image size to 12x12 pixels
    pool1 = max_pool(relu1)
    
    # Apply batch normalization
    batch1 = batch_norm(pool1, 64, is_training)
    
# Second convolutional layer

with tf.name_scope('second_convolutional_layer'):
    W_conv2 = weight_variable([3, 3, 64, 96], 'W_conv2', False)
    b_conv2 = bias_variable([96], 'b_conv2')
    
    conv2 = convolve(batch1, W_conv2) + b_conv2
    relu2 = tf.nn.relu(conv2)
    batch2 = batch_norm(relu2, 96, is_training)
    pool2 = max_pool(batch2)
    
# First fully connected layer
    
with tf.name_scope('first_fully_connected_layer'):
    # Flatten the tensor into 1 dimension
    pool2_flat = tf.reshape(pool2, [-1, 6 * 6 * 96])
    
    # Prepare the network variables
    W_conn1 = weight_variable([6 * 6 * 96, 384], 'W_conn1', True)
    b_conn1 = bias_variable([384], 'b_conn1')
    
    conn1 = tf.matmul(pool2_flat, W_conn1) + b_conn1
    relu3 = tf.nn.relu(conn1)
    
# First dropout layer

with tf.name_scope('first_dropout_layer'):
    drop1 = tf.nn.dropout(relu3, keep)
    
# Second fully connected layer

with tf.name_scope('second_fully_connected_layer'):
    W_conn2 = weight_variable([384, 192], 'W_conn2', True)
    b_conn2 = bias_variable([192], 'b_conn2')
    
    conn2 = tf.matmul(drop1, W_conn2) + b_conn2
    relu4 = tf.nn.relu(conn2)
    
# Second dropout layer

with tf.name_scope('second_dropout_layer'):
    drop2 = tf.nn.dropout(relu4, keep)
    
# Output layer

with tf.name_scope('third_fully_connected_layer'):
    W_conn3 = weight_variable([192, 10], 'W_conn3', False)
    b_conn3 = bias_variable([10], 'b_conn3')
    
    conn3 = tf.matmul(drop2, W_conn3) + b_conn3
    
''' Additional functions and ops '''

# Cost function
with tf.name_scope('cost_function'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_actual, logits = conn3))
    tf.summary.scalar('cost_function', cross_entropy)

# Train step
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Accuracy
with tf.name_scope('network_accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(conn3, 1), tf.argmax(y_actual, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('training_accuracy', accuracy)

# Initialization op
init_op = tf.global_variables_initializer()

# Save and restore op
saver = tf.train.Saver()

# Save locations
save_location = args.save_dir + '/cobalt.ckpt'
log_directory = args.log_dir

# Download the data set
download_data_set()

''' Prepare the data '''

# Load the data
images_train_raw, classes_train, labels_train = load_training_data()
images_test_raw, classes_test, labels_test = load_test_data()

# Process the images
with tf.Session() as proc_sess:
    if args.steps != 0:
        with tf.name_scope('training_image_processing'):
            images_train = process_images(images_train_raw, args.data_augmentation).eval()
    if args.steps != 0 or args.accuracy:
        with tf.name_scope('validation_image_processing'):
            images_test = process_images(images_test_raw, False).eval()
        
''' Train the network '''
    
# Start the session
with tf.Session() as sess:
    sess.run(init_op)

    if args.steps != 0 and (glob.glob(save_location + '*') == [] or args.overwrite):

        # Merge all summaries
        merged = tf.summary.merge_all()

        # Summary writers
        train_writer = tf.summary.FileWriter(log_directory + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(log_directory + '/validation')

        # Validation accuracy summary
        accuracy_summary = tf.summary.scalar('validation_accuracy', accuracy)
        
        # Run the training loop, show progress every 1000th step
        for i in range(args.steps):

            # Decay the learning rate
            learn = initial_learning_rate * learning_rate_decay ** (i // learning_decay_frequency)
            
            x_batch, y_actual_batch = random_batch()
            feed_dict_train = {x: x_batch, y_actual: y_actual_batch, keep: 0.5, is_training: True, learning_rate: learn}

            # Execute a training step`
            summary, _ = sess.run([merged, train_step], feed_dict_train)

            # Write a summary
            train_writer.add_summary(summary, i)
            
            if i % 100 == 0:
                x_test, y_actual_test = random_batch(True)
                feed_dict_validation = {x: x_test, y_actual: y_actual_test, keep: 1, is_training: False}
                
                summary, validation_accuracy = sess.run([accuracy_summary, accuracy], feed_dict_validation)
                validation_writer.add_summary(summary, i)
                
                if i % 1000 == 0:
                    print('Training network (step %g/%g), current accuracy: %g' % (i, args.steps, validation_accuracy))

        # Save the network variables to disk
        saver.save(sess, save_location)
        print('Network saved to %s' % save_location)

    elif args.steps != 0:
        print('Found saved network at %s, pick a new save location or use --overwrite' % save_location)
        sess.close()
        exit()
        
    if args.accuracy and glob.glob(save_location + '*') != []:
        # Load the old network variables
        saver.restore(sess, save_location)
        print('Network loaded from %s' % save_location)
            
        # Get final accuracy
        feed_dict_final = {x: images_test, y_actual: labels_test, keep: 1, is_training: False}
        sys.stdout.write('Getting accuracy...')
        sys.stdout.flush()
        print('\rNetwork accuracy: %g' % accuracy.eval(feed_dict_final))
    elif args.accuracy:
        print('Could not find saved network, unable to check accuracy')
