#!/usr/bin/env python

import os
import sys

_root_dir = os.path.dirname(__file__)
sys.path.append(_root_dir)

import numpy as np

from phishing.classifier.neural_net import NeuralNetwork
from phishing.classifier.nn_layer.perceptron import Perceptron

# training data
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# net = NeuralNetwork()
# net.add(Perceptron(2, 3))
# net.add(Perceptron(3, 1))

net = NeuralNetwork(
    Perceptron(2, 3),
    Perceptron(3, 1)
)

net.fit(x_train, y_train)
prediction = net.predict(x_train)

print(prediction.shape, prediction)
