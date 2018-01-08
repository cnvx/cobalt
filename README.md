# cobalt

Convolutional neural network for image based object classification.

## About

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

I found Hvass-Labs and his [excellent repository](https://github.com/Hvass-Labs/TensorFlow-Tutorials) very helpful (although most of it uses the simplified [Pretty Tensor API](https://github.com/google/prettytensor). In particular the image preprocessing code was copied almost word for word.

A thanks to Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton for collecting the images used in and maintaining the [CIFAR-10 data set](https://www.cs.toronto.edu/~kriz/cifar.html).

## Disclaimer

This is **alpha** (and mostly broken) code!

## License

The [MIT License](LICENSE).
