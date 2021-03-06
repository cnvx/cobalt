**This project has been moved to https://gitlab.com/archiekennedy/cobalt, consider this repository outdated.**

# Cobalt Neural Network

Convolutional residual neural network for image-based object classification, used in [this Android app](https://github.com/cnvx/argon).

## About

Based on the [WRN architecture](https://arxiv.org/pdf/1605.07146.pdf), training and validation data taken from the [CIFAR-100 data set](https://www.cs.toronto.edu/~kriz/cifar.html).  
Capable of reaching about 74% accuracy after 80000 training steps (just over 22 hours with a GTX 1070 8GB).

If you run out of memory try using a smaller batch size, for best results use [powers of 2](https://en.wikipedia.org/wiki/Power_of_two).

## Usage

Linux (or anything that supports [shebangs](https://en.wikipedia.org/wiki/Shebang_(Unix))):  
`./cobalt.py --help`

Windows:  
`python cobalt.py --help`

### Examples

`./cobalt.py --train 40000`  
`./cobalt.py --accuracy`  
`./cobalt.py -t 80000 --export network`

### Visualizing

To represent training graphically:

1. Navigate to the cobalt directory.
2. Run `tensorboard --logdir=data/log` or `python -m tensorboard.main --logdir=data/log` on Windows (replace `data` with whatever you set using `--save`).
3. Open [localhost:6006](http://localhost:6006/) in your browser.

## Requirements

Make sure you have a 64-bit operating system.

### Linux

If you're running something Debian based:

```
sudo apt-get update
sudo apt-get install python3-pip python3-dev python3-tk
sudo python3 -m pip install -U setuptools
sudo python3 -m pip install -U tensorflow numpy
```

### Linux with GPU support

Training is a lot faster with a graphics card, but getting it running is an involved process:

1. `sudo apt-get update && sudo apt-get install python3-pip python3-dev python3-tk`
2. Install CUDA Toolkit **9.0** from [here](https://developer.nvidia.com/cuda-zone) by following [this guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
3. Get the cuDNN SDK 7.0.5 **for CUDA 9.0** from [here](https://developer.nvidia.com/rdp/cudnn-archive), follow the *cuDNN Install Guide* PDF for that specific version.
4. `sudo apt-mark hold libcudnn7 libcudnn7-dev`
5. `sudo pip3 install -U setuptools && sudo python3 -m pip install -U tensorflow-gpu`

### Windows

1. Get Python from [here](https://www.python.org/downloads/release/python-365/) (download the x86-64 version), make sure to tick *Add Python 3.6 to PATH* during installation.
2. Download [this file](https://bootstrap.pypa.io/get-pip.py) and run it with `python get-pip.py` to install pip.
3. `python -m pip install -U setuptools`
4. `python -m pip install -U tensorflow numpy`

## Deploy

To use this with your own code, save the trained network as a [.pb file](https://developers.google.com/protocol-buffers/) with `--export`.

### Android

If you use Java have a look at [this class](https://github.com/cnvx/argon/blob/master/app/src/main/java/com/example/cnvx/argon/CobaltClassifier.java) from my app for an example on using the [TensorFlowInferenceInterface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/android/java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java), or read [this tutorial](https://medium.com/capital-one-developers/using-a-pre-trained-tensorflow-model-on-android-e747831a3d6).

### Python

TensorFlow offers the [SavedModel loader](https://www.tensorflow.org/api_docs/python/tf/saved_model/loader), read about it [here](https://www.tensorflow.org/programmers_guide/saved_model#apis_to_build_and_load_a_savedmodel).

### Something else

Protocol Buffers support a few [other languages](https://developers.google.com/protocol-buffers/docs/tutorials) and people are working on adding [more](https://github.com/google/protobuf/blob/master/docs/third_party.md).

## Acknowledgements

[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (Alex Krizhevsky, Ilya Sutskever & Geoffrey Hinton, 2012).  
[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) (Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever & Ruslan Salakhutdinov, 2014).  
[Wide Residual Networks](https://arxiv.org/pdf/1605.07146.pdf) (Sergey Zagoruyko & Nikos Komodakis, 2016).  
[A Method for Solving a Convex Programming Problem with Convergence Rate O(1/K2)](http://mpawankumar.info/teaching/cdt-big-data/nesterov83.pdf) (Yurii Nesterov, 1983).  
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) (Kaiming He, Xiangyu Zhang, Shaoqing Ren & Jian Sun, 2015).  
[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v3.pdf) (Sergey Ioffe & Christian Szegedy, 2015).  
Magnus Pedersen and his [excellent repository](https://github.com/Hvass-Labs/TensorFlow-Tutorials).  
A thanks to Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton for maintaining the [CIFAR-100 data set](https://www.cs.toronto.edu/~kriz/cifar.html).  
[Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) (Alex Krizhevsky, 2009).

## License

This repository uses the [MIT License](LICENSE).  
TensorFlow is released under the [Apache License, Version 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).

## Architecture
![Neural network layers](layers.png)