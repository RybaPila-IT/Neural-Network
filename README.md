# Project Overview
The Neural-Network project is the implementation of 
artificial neural network (aka multi-layer perceptron). <br>
Project is my "Hello world" in AI field.

## Network description
Network implemented in network.py module is fully-functional 
artificial neural network. <br>

Network:
- Supports L2 regularization.
- Monitors learning process, adjusting eta learning rate if network
stops to make progress.
- Enables to load and store notwork in json file format.  

Network implementation uses:
- Regular stochastic gradient descent as the learning algorithm.
- Fully vectorized forward-and-backpropagation written in numpy.
- Matplotlib library to plot graphs useful for monitoring network learning progress.

Network constructor:
- Enables specifying activation function (default used is sigmoid function).
- Enables specifying cost function (default is cross entropy cost function).
- Enables specifying metric for measuring progress on validation set (default is accuracy metric).

## MNIST Dataset
Network performance is presented with usage of MNIST Dataset.
MNIST Dataset is a set of hand-written digits with corresponding
output labels. <br>
Module loader.py includes functions loading dataset and returning 
results as numpy arrays.

## Training results
Results of training network are stored in file net.json. 
Network (architecture: [784, 100, 30, 10]; training: eta=0.15, lambda_r=5, epochs=150)
achieved a bit more than 97% accuracy on validation and test dataset.

## Resources
Project was highly inspired by 3blue1brown youtube channel
(series about neural networks) and neuralnetworksanddeeplearning
(I am bound to state that without this book and provided
python code this project will not be finished).
