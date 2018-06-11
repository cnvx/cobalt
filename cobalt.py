#!/usr/bin/env python3

__version__ = '2.1.0'

import tensorflow as tf
import numpy as np
import os
import sys
import urllib.request
import tarfile
import pickle
import glob
import shutil
import hashlib
import argparse as arg

''' Argument parsing '''

parser_format = lambda prog: arg.HelpFormatter(prog, max_help_position = 79)
parser = arg.ArgumentParser(formatter_class = parser_format)

parser.add_argument('-t', '--train', metavar = 'steps', dest = 'times_to_train', type = int,
                    default = 0, help = 'how many times to train')
parser.add_argument('-o', '--overwrite', action = 'store_true', default = False,
                    help = 'overwrite saved network data')
parser.add_argument('-b', '--batch', metavar = 'size', dest = 'batch_size', type = int,
                    default = 128, help = 'batch size used during training')
parser.add_argument('-a', '--accuracy', action = 'store_true', default = False,
                    help = 'check network validation accuracy')
parser.add_argument('--accuracy-batch', metavar = 'size', dest = 'accuracy_batch_size', type = int,
                    default = 1000, help = 'batch size used during accuracy calculation')
parser.add_argument('-s', '--save', metavar = 'directory', dest = 'save_dir', default = 'data',
                    help = 'logs and trained network save location')
parser.add_argument('-e', '--export', metavar = 'name', dest = 'export',
                    help = 'export language-neutral network')

ascii_art = '''
     _/_/_/                   _/      _/       _/      _/
  _/              _/_/       _/_/    _/       _/_/    _/
 _/            _/    _/     _/  _/  _/       _/  _/  _/
_/            _/    _/     _/    _/_/       _/    _/_/
 _/_/_/        _/_/       _/      _/       _/      _/

Cobalt Neural Network version {}
_________________________________________________________
'''.format(__version__)

if __name__ == '__main__':
    print(ascii_art)

''' Hyperparameters '''

initial_learning_rate = 0.01
learning_rate_decay = 0.92
learning_decay_frequency = 10000
momentum = 0.9
initial_weight_decay = 5e2
moving_average_decay = 0.5
data_augmentation = True
widening_factor = 12

''' Functions for getting the CIFAR-100 data set '''

data_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
data_hash = 'eb9058c3a382ffc7106e4002c42a8d85'
data_path = './CIFAR-100/'

# Image dimensions
image_size = 32

# Colour channels
number_of_channels = 3

# Possible objects
number_of_classes = 100

# Total number of images
number_of_images_train = number_of_classes * 500
number_of_images_test = number_of_classes * 100

# Get full file path, return directory if called with no filename
def get_file_path(filename = ''):
    return os.path.join(data_path, 'cifar-100-python', filename)
    
def unpickle(filename):
    file_path = get_file_path(filename)
    print('Loading data from ' + file_path)

    with open(file_path, mode = 'rb') as file:
        data = pickle.load(file, encoding = 'bytes')

    return data

# Convert from CIFAR-100 format to 4 dimensional array
def convert_images(unconverted):
    float_array = np.array(unconverted, dtype = float) / 255.0

    # Reshape to 4 dimensions
    images = float_array.reshape([-1, number_of_channels, image_size, image_size])
    images = images.transpose([0, 2, 3, 1])

    return images

# Return a one hot encoded 2 dimensional array of class numbers and labels
def one_hot_encoded(class_numbers, number_of_classes):
    return np.eye(number_of_classes, dtype = float)[class_numbers]

# Load pickled data and return converted images and their class numbers
def load_data(filename):
    data = unpickle(filename)

    images = convert_images(data[b'data'])
    class_numbers = np.array(data[b'fine_labels'])

    return images, one_hot_encoded(class_numbers, number_of_classes)

# Download and extract the data set if it doesn't already exist
def download_data_set():
    # Output the download progress
    def print_progress(count, block_size, total_size):
        percentage = float(count * block_size) / total_size
        message = '\rDownloading data set... {0:.1%}'.format(percentage)

        sys.stdout.write(message)
        sys.stdout.flush()

    # Add the filename from the URL to the download_directory
    filename = data_url.split('/')[-1]
    file_path = os.path.join(data_path, filename)

    # Verify the integrity of the archive
    if not os.path.exists(file_path) or hashlib.md5(open(file_path, 'rb').read()).hexdigest() != data_hash:
        # Delete the download directory if it exists
        if os.path.exists(data_path):
            shutil.rmtree(data_path, ignore_errors = True)

        os.makedirs(data_path)

        # Download the data set
        file_path, _ = urllib.request.urlretrieve(url = data_url,
                                                  filename = file_path,
                                                  reporthook = print_progress)

        sys.stdout.write('\rDownload complete             \nExtracting files...')
        sys.stdout.flush()
        
        tarfile.open(name = file_path, mode = 'r:gz').extractall(data_path)

        print('\rFiles extracted    ')
    else:
        print('Data has already been downloaded and extracted')

''' Variables and placeholders for the neural network '''

# Input images for the neural network
x = tf.placeholder(tf.float32, shape = [None, image_size, image_size, number_of_channels], name = 'x')

# Real lables associated with each image
y_actual = tf.placeholder(tf.float32, shape = [None, number_of_classes], name = 'y_actual')

is_training = tf.placeholder_with_default(False, shape = (), name = 'is_training')
augment = tf.placeholder_with_default(False, shape = (), name = 'augment')

# Learning rate
with tf.name_scope('learning_rate'):
    learning_rate = tf.placeholder(tf.float32, shape = [])
    tf.summary.scalar('learning_rate', learning_rate)

''' Image processing functions '''

# If data augmentation is enabled make random modifications
def process_single_image(image, augment):
    if augment:
        image = tf.pad(image, [[2, 2], [2, 2], [0, 0]], 'SYMMETRIC')
        image = tf.random_crop(image, size = [image_size, image_size, number_of_channels])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, max_delta = 0.05)
        image = tf.image.random_contrast(image, lower = 0.3, upper = 1.0)
        image = tf.image.random_saturation(image, lower = 0.0, upper = 2.0)
        image = tf.image.random_brightness(image, max_delta = 0.2)

        # Stop tf.image.random_contrast() from outputting extreme values
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)

    return image

def process_images(images, augment):
    images = tf.map_fn(lambda image: process_single_image(image, augment), images)
    return images

# Data augmentation
with tf.name_scope('data_augmentation'):
    with tf.device('/cpu:0'):
        proc = tf.cond(augment, lambda: process_images(x, True), lambda: process_images(x, False))

# Get random batch of images and labels
def random_batch(images, labels, batch_size, validation = False):
    if validation:
        random = np.random.choice(number_of_images_test, size = batch_size, replace = False)
    else:
        random = np.random.choice(number_of_images_train, size = batch_size, replace = False)

    x_batch = images[random, :, :, :]
    y_batch = labels[random, :]
    
    return x_batch, y_batch

''' Functions for creating the neural network '''

# Create weight variable using Xavier initialisation
def weight_variable(shape, decay, name):
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

# Convolutional layer
def conv_layer(x, filter_size, stride, in_filters, out_filters, layer):
    W = weight_variable([filter_size, filter_size, in_filters, out_filters], True, 'weight_' + layer)
    b = bias_variable([out_filters], 'bias_' + layer)

    convolved = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'SAME')

    return convolved + b

# Batch normalisation
def batch_norm_layer(x, depth, is_training):
    mean_batch, variance_batch = tf.nn.moments(x, [0, 1, 2], name = 'batch_mean_variance_calculation')
    moving_average = tf.train.ExponentialMovingAverage(decay = moving_average_decay)
    
    def mean_variance_update():
        moving_average_op = moving_average.apply([mean_batch, variance_batch])
        with tf.control_dependencies([moving_average_op]):
            return tf.identity(mean_batch), tf.identity(variance_batch)
        
    mean, variance = tf.cond(is_training, mean_variance_update, lambda: (moving_average.average(mean_batch),
                                                                         moving_average.average(variance_batch)))

    # Also known as offset and scale
    beta = tf.Variable(tf.constant(0.0, shape = [depth]), name = 'beta', trainable = True)
    gamma = tf.Variable(tf.constant(1.0, shape = [depth]), name = 'gamma', trainable = True)
    
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-4)

''' Neural network architecture '''

# Input block
# Input: ?x32x32x3
# Output: ?x32x32x16
with tf.name_scope('input_block'):
    conv1 = conv_layer(x, 3, 1, number_of_channels, 16, 'conv1')
    batch1 = batch_norm_layer(conv1, 16, is_training)
    relu1 = tf.nn.relu(batch1)

# First residual block
# Output: ?x32x32x192
with tf.name_scope('first_residual_block'):
    conv2 = conv_layer(relu1, 3, 1, 16, 16 * widening_factor, 'conv2')
    batch2 = batch_norm_layer(conv2, 16 * widening_factor, is_training)
    relu2 = tf.nn.relu(batch2)
    conv3 = conv_layer(relu2, 3, 1, 16 * widening_factor, 16 * widening_factor, 'conv3')
    batch3 = batch_norm_layer(conv3, 16 * widening_factor, is_training)
    relu3 = tf.nn.relu(batch3)

    # Shortcut connection with the previous block
    short1 = relu3 + conv_layer(relu1, 1, 1, 16, 16 * widening_factor, 'short1')

# Second residual block
# Output: ?x32x32x192
with tf.name_scope('second_residual_block'):
    conv4 = conv_layer(short1, 3, 1, 16 * widening_factor, 16 * widening_factor, 'conv4')
    batch4 = batch_norm_layer(conv4, 16 * widening_factor, is_training)
    relu4 = tf.nn.relu(batch4)
    conv5 = conv_layer(relu4, 3, 1, 16 * widening_factor, 16 * widening_factor, 'conv5')
    batch5 = batch_norm_layer(conv5, 16 * widening_factor, is_training)
    relu5 = tf.nn.relu(batch5)

    short2 = relu5 + short1

# Third residual block
# Output: ?x32x32x192
with tf.name_scope('third_residual_block'):
    conv6 = conv_layer(short2, 3, 1, 16 * widening_factor, 16 * widening_factor, 'conv6')
    batch6 = batch_norm_layer(conv6, 16 * widening_factor, is_training)
    relu6 = tf.nn.relu(batch6)
    conv7 = conv_layer(relu6, 3, 1, 16 * widening_factor, 16 * widening_factor, 'conv7')
    batch7 = batch_norm_layer(conv7, 16 * widening_factor, is_training)
    relu7 = tf.nn.relu(batch7)

    short3 = relu7 + short2

# Fourth residual block
# Output: ?x16x16x384
with tf.name_scope('fourth_residual_block'):
    conv8 = conv_layer(short3, 3, 2, 16 * widening_factor, 32 * widening_factor, 'conv8')
    batch8 = batch_norm_layer(conv8, 32 * widening_factor, is_training)
    relu8 = tf.nn.relu(batch8)
    conv9 = conv_layer(relu8, 3, 1, 32 * widening_factor, 32 * widening_factor, 'conv9')
    batch9 = batch_norm_layer(conv9, 32 * widening_factor, is_training)
    relu9 = tf.nn.relu(batch9)

    short4 = relu9 + conv_layer(short3, 1, 2, 16 * widening_factor, 32 * widening_factor, 'short4')

# Fifth residual block
# Output: ?x16x16x384
with tf.name_scope('fifth_residual_block'):
    conv10 = conv_layer(short4, 3, 1, 32 * widening_factor, 32 * widening_factor, 'conv10')
    batch10 = batch_norm_layer(conv10, 32 * widening_factor, is_training)
    relu10 = tf.nn.relu(batch10)
    conv11 = conv_layer(relu10, 3, 1, 32 * widening_factor, 32 * widening_factor, 'conv11')
    batch11 = batch_norm_layer(conv11, 32 * widening_factor, is_training)
    relu11 = tf.nn.relu(batch11)

    short5 = relu11 + short4

# Sixth residual block
# Output: ?x16x16x384
with tf.name_scope('sixth_residual_block'):
    conv12 = conv_layer(short5, 3, 1, 32 * widening_factor, 32 * widening_factor, 'conv12')
    batch12 = batch_norm_layer(conv12, 32 * widening_factor, is_training)
    relu12 = tf.nn.relu(batch12)
    conv13 = conv_layer(relu12, 3, 1, 32 * widening_factor, 32 * widening_factor, 'conv13')
    batch13 = batch_norm_layer(conv13, 32 * widening_factor, is_training)
    relu13 = tf.nn.relu(batch13)

    short6 = relu13 + short5

# Seventh residual block
# Output: ?x8x8x768
with tf.name_scope('seventh_residual_block'):
    conv14 = conv_layer(short6, 3, 2, 32 * widening_factor, 64 * widening_factor, 'conv14')
    batch14 = batch_norm_layer(conv14, 64 * widening_factor, is_training)
    relu14 = tf.nn.relu(batch14)
    conv15 = conv_layer(relu14, 3, 1, 64 * widening_factor, 64 * widening_factor, 'conv15')
    batch15 = batch_norm_layer(conv15, 64 * widening_factor, is_training)
    relu15 = tf.nn.relu(batch15)

    short7 = relu15 + conv_layer(short6, 1, 2, 32 * widening_factor, 64 * widening_factor, 'short7')

# Eighth residual block
# Output: ?x8x8x768
with tf.name_scope('eighth_residual_block'):
    conv16 = conv_layer(short7, 3, 1, 64 * widening_factor, 64 * widening_factor, 'conv16')
    batch16 = batch_norm_layer(conv16, 64 * widening_factor, is_training)
    relu16 = tf.nn.relu(batch16)
    conv17 = conv_layer(relu16, 3, 1, 64 * widening_factor, 64 * widening_factor, 'conv17')
    batch17 = batch_norm_layer(conv17, 64 * widening_factor, is_training)
    relu17 = tf.nn.relu(batch17)

    short8 = relu17 + short7

# Ninth residual block
# Output: ?x8x8x768
with tf.name_scope('ninth_residual_block'):
    conv18 = conv_layer(short8, 3, 1, 64 * widening_factor, 64 * widening_factor, 'conv18')
    batch18 = batch_norm_layer(conv18, 64 * widening_factor, is_training)
    relu18 = tf.nn.relu(batch18)
    conv19 = conv_layer(relu18, 3, 1, 64 * widening_factor, 64 * widening_factor, 'conv19')
    batch19 = batch_norm_layer(conv19, 64 * widening_factor, is_training)
    relu19 = tf.nn.relu(batch19)

    short9 = relu19 + short8

# Global average pooling layer
# Output: ?x1x1x768
with tf.name_scope('global_average_pooling_layer'):
    avg_pool = tf.nn.pool(short9, [8, 8], 'AVG', padding = 'VALID')

# Fully connected layer
# Output: ?x100
with tf.name_scope('fully_connected_layer'):
    # Flatten the tensor into 1 dimension
    flat = tf.reshape(avg_pool, [-1, 1 * 1 * 768])

    # Prepare the network variables
    W_conn = weight_variable([1 * 1 * 768, number_of_classes], True, 'weight_conn')
    b_conn = bias_variable([number_of_classes], 'bias_conn')

    conn = tf.matmul(flat, W_conn) + b_conn
    soft = tf.nn.softmax(conn, name = 'output')

''' Additional functions and operations '''

# Cost function
with tf.name_scope('cost_function'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = conn, labels = y_actual))
    tf.summary.scalar('cost_function', cross_entropy)

# Train step
with tf.name_scope('train'):
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov = True).minimize(cross_entropy)

# Accuracy
with tf.name_scope('network_accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(soft, 1), tf.argmax(y_actual, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('training_accuracy', accuracy)

# Initialisation op
init_op = tf.global_variables_initializer()

# Save and restore op
saver = tf.train.Saver()

def main():
    args = parser.parse_args()

    # Save locations
    save_location = os.path.join(args.save_dir, 'cobalt.ckpt')
    log_directory = os.path.join(args.save_dir, 'log')

    # Display help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        return

    if args.times_to_train > 0 or args.accuracy == True or args.accuracy_batch_size != 1000:
        download_data_set()

        images_train, labels_train = load_data('train')
        images_test, labels_test = load_data('test')
    elif args.times_to_train < 0:
        print('Training steps must not be negative, specify a new number with --train')

    ''' Create the neural network '''

    # Start the session
    with tf.Session() as sess:
        sess.run(init_op)

        if args.times_to_train > 0 and (glob.glob(save_location + '*') == [] or args.overwrite):
            if data_augmentation:
                data_augmentation_status = 'enabled'
            else:
                data_augmentation_status = 'disabled'

            print('Initialising neural network with the following hyperparameters:\n'
                  '  Initial learning rate: {}\n'
                  '  Learning rate decay: {}\n'
                  '  Learning rate decay frequency: {}\n'
                  '  Momentum: {}\n'
                  '  Initial weight decay: {:g}\n'
                  '  Exponential moving average decay: {}\n'
                  '  Data augmentation: {}\n'
                  '  Widening factor: {}\n'
                  '  Batch size: {}'
                  .format(initial_learning_rate, learning_rate_decay, learning_decay_frequency, momentum,
                          initial_weight_decay, moving_average_decay, data_augmentation_status, widening_factor,
                          args.batch_size))

            # Delete the old logs if they exist
            if glob.glob(log_directory):
                shutil.rmtree(log_directory, ignore_errors = True)

            # Merge all summaries
            merged = tf.summary.merge_all()

            # Summary writers
            train_writer = tf.summary.FileWriter(log_directory + '/train', sess.graph)
            validation_writer = tf.summary.FileWriter(log_directory + '/validation')

            # Validation accuracy summary
            accuracy_summary = tf.summary.scalar('validation_accuracy', accuracy)

            # Run the training loop, show progress every 1000th step
            for i in range(args.times_to_train):
                # Decay the learning rate
                learn = initial_learning_rate * learning_rate_decay ** (i // learning_decay_frequency)

                # Perform data augmentation
                x_train, y_actual_train = random_batch(images_train, labels_train, args.batch_size)
                augmented = sess.run(proc, {x: x_train, augment: data_augmentation})

                feed_dict_train = {x: augmented, y_actual: y_actual_train, is_training: True, learning_rate: learn}

                # Execute a training step
                summary, _ = sess.run([merged, train_step], feed_dict_train)

                # Write a summary
                train_writer.add_summary(summary, i)

                if i % 100 == 0:
                    x_test, y_actual_test = random_batch(images_test, labels_test, args.batch_size, True)
                    feed_dict_validation = {x: x_test, y_actual: y_actual_test, is_training: False}

                    # Get the current validation accuracy
                    summary, validation_accuracy = sess.run([accuracy_summary, accuracy], feed_dict_validation)
                    validation_writer.add_summary(summary, i)

                    if i % 1000 == 0:
                        print('Training network (step {}/{}), current accuracy: {}%'
                              .format(i, args.times_to_train, round(validation_accuracy * 100, 2)))

            # Save the network variables to disk
            saver.save(sess, save_location)
            print('Network saved to {}'.format(save_location))

        elif args.times_to_train > 0:
            print('Found saved network in {}, pick a new save location or use --overwrite'.format(save_location))

        if args.accuracy or args.accuracy_batch_size != 1000 and glob.glob(save_location + '*') != []:
            if args.accuracy_batch_size <= number_of_images_test:
                # Load the old network variables
                saver.restore(sess, save_location)
                print('Network loaded from {}'.format(save_location))

                x_final, y_actual_final = random_batch(images_test, labels_test, args.accuracy_batch_size, True)
                feed_dict_final = {x: x_final, y_actual: y_actual_final, is_training: False}

                # Get final accuracy
                print('Calculating accuracy based on {} validation images, increase this with --accuracy-batch'
                      .format(args.accuracy_batch_size))
                print('Network accuracy: {}%'.format(round(accuracy.eval(feed_dict_final) * 100, 2)))
            else:
                print('Batch size exceeds total number of validation images ({}), '
                      'specify a smaller size with --accuracy-batch'
                      .format(number_of_images_test))
        elif args.accuracy or args.accuracy_batch_size != 1000:
            print('Could not find saved network, unable to check accuracy')

    ''' Export network for inference, using protocol buffers '''

    if args.export is not None:

        # Set filename extension
        export_file = args.export
        if export_file.endswith('.pb') == False:
            export_file = export_file + '.pb'

        with tf.Session(graph = tf.Graph()) as export_sess:
            # Load the saved network
            meta_saver = tf.train.import_meta_graph(save_location + '.meta', clear_devices = True)
            meta_saver.restore(export_sess, save_location)

            graph_data = export_sess.graph.as_graph_def()

            # Fix bug related to batch normalisation
            for node in graph_data.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr:
                        del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    node.op = 'Add'
                    if 'use_locking' in node.attr:
                        del node.attr['use_locking']

            network = tf.graph_util.convert_variables_to_constants(export_sess, graph_data,
                                                                   ['fully_connected_layer/output'])

            # Export the network using protocol buffers
            with tf.gfile.GFile(export_file, 'wb') as exporter:
                exporter.write(network.SerializeToString())

            print('Neural network for inference exported to {}'.format(export_file))

if __name__ == '__main__':
    main()
