# cobalt

Convolutional neural network for image-based object classification. Used in [this Android app](https://github.com/cnvx/argon).

## About

Based on the [AlexNet architecture](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), training and validation data taken from the [CIFAR-10 data set](https://www.cs.toronto.edu/~kriz/cifar.html).  
Capable of reaching about 84% accuracy after 175000 training steps.

If you run out of memory try using a smaller batch size (256 by default).

### Usage

Linux (or anything that supports [shebangs](https://en.wikipedia.org/wiki/Shebang_(Unix))):  
`./cobalt.py --help`

Windows:  
`python cobalt.py --help`

### Examples

`./cobalt.py --train 20000`  
`./cobalt.py --accuracy`  
`./cobalt.py -t 50000 --export network`

### Visualizing

To represent training graphically:

1. Navigate to the cobalt directory.
2. Run `tensorboard --logdir=data/log` or `python -m tensorboard.main --logdir=data/log` on Windows (replace `data` with whatever you set using `--save`).
3. Open [localhost:6006](http://localhost:6006/) in your browser.

### Requirements

#### Linux

If you're running something Debian based:

```
sudo apt-get update
sudo apt-get install python3-pip python3-dev python3-tk
sudo pip3 install -U setuptools
sudo pip3 install tensorflow numpy
```

#### Windows

1. Get Python from [here](https://www.python.org/downloads/release/python-362/), make sure to tick *Add Python 3.x to PATH* during installation.
2. Download and run [this file](https://bootstrap.pypa.io/get-pip.py) to install pip.
3. `python -m pip install --user tensorflow numpy`

## Deploy

To use this with your own code, save the trained network as a [.pb file](https://developers.google.com/protocol-buffers/) with `--export`.

### Android

If you use Java have a look at [this class](https://github.com/cnvx/argon/blob/master/app/src/main/java/com/example/cnvx/argon/CobaltClassifier.java) from my app for an example on using the [TensorFlowInferenceInterface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/android/java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java), or read [this tutorial](https://medium.com/capital-one-developers/using-a-pre-trained-tensorflow-model-on-android-e747831a3d6).

### Python

TensorFlow offers the [SavedModel loader](https://www.tensorflow.org/api_docs/python/tf/saved_model/loader), read about it [here](https://www.tensorflow.org/programmers_guide/saved_model#apis_to_build_and_load_a_savedmodel).

### Something else

Protocol Buffers support a few [other languages](https://developers.google.com/protocol-buffers/docs/tutorials) and people are working on adding [more](https://github.com/google/protobuf/blob/master/docs/third_party.md).

## Acknowledgements

If you're interested in machine learning I recommend starting with [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) by Michael Nielsen.  
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton, 2012).  
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v3.pdf) (Sergey Ioffe & Christian Szegedy, 2015).  
Magnus Pedersen and his [excellent repository](https://github.com/Hvass-Labs/TensorFlow-Tutorials), in particular his download and data augmentation code.  
A thanks to Alex Krizhevsky, Vinod Nair, and Geoffrey E. Hinton for creating and maintaining the [CIFAR-10 data set](https://www.cs.toronto.edu/~kriz/cifar.html).  
[Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) (Alex Krizhevsky, 2009).

## License

This repository uses the [MIT License](LICENSE).  
Data set download and augmentation functions belong to [Magnus Pedersen](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/LICENSE).  
TensorFlow is released under the [Apache License, Version 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).
