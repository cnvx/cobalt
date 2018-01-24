# cobalt

Convolutional neural network for image based object classification.

## About

Use a GPU and lots of memory for best results.  
Training and validation data taken from the [CIFAR-10 data set](https://www.cs.toronto.edu/~kriz/cifar.html).

### Usage

Linux (or anything that supports [shebangs](https://en.wikipedia.org/wiki/Shebang_(Unix))):  
`./cobalt.py [overwrite_network(y/n)] [times_to_train]`

Windows:  
`python cobalt.py [overwrite_network(y/n)] [times_to_train]`

### Examples

`./cobalt.py`  
`./cobalt.py n`  
`./cobalt.py y 50000`

### Visualizing

To represent training graphically:

1. Navigate to the cobalt directory.
2. Run `tensorboard --logdir=log` or `python -m tensorboard.main --logdir=log` on Windows.
3. Open [localhost:6006](http://localhost:6006/) in your browser.

## Requirements

### Linux

If you're running something Debian based:

```
sudo apt-get update
sudo apt-get install python3-pip python3-dev python3-tk
sudo pip3 install -U setuptools
sudo pip3 install tensorflow numpy
```

### Windows

1. Get Python from [here](https://www.python.org/downloads/release/python-362/), make sure to tick *Add Python 3.x to PATH* during installation.
2. Download and run [this file](https://bootstrap.pypa.io/get-pip.py) to install pip.
3. `python -m pip install --user tensorflow numpy`

## Acknowledgements

If you're interested in machine learning I recommend reading [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) by Michael Nielsen.

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167v3.pdf) (Sergey Ioffe & Christian Szegedy, 2015).

Hvass-Labs and his [excellent repository](https://github.com/Hvass-Labs/TensorFlow-Tutorials), in particular his download and data augmentation code.

A thanks to Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton for creating and maintaining the [CIFAR-10 data set](https://www.cs.toronto.edu/~kriz/cifar.html).  
[Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) (Alex Krizhevsky, 2009).

## Disclaimer

This is **alpha** code! I wouldn't use it for anything important.

## License

The [MIT License](LICENSE).
