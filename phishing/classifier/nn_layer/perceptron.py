"""Perceptron Layer Module

This module implements a fully-connected perceptron layer for use with the
`NeuralNetwork` class.
"""

import numpy as np

from .base import Layer
from ..nn_utils.activation import sigmoid, d_sigmoid


class Perceptron(Layer):
    def __init__(
        self, inputs, outputs, *, activation=sigmoid, dActivation=d_sigmoid
    ) -> None:
        super().__init__(inputs, outputs)

        self.weights = np.random.rand(inputs, outputs) - 0.5
        self.bias = np.random.rand(1, outputs) - 0.5
        self.activation = activation
        self.dActivation = dActivation

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias

        return self.activation(self.output)

    def backward(self, output_err, alpha):
        output_err = self.dActivation(self.output) * output_err
        input_err = np.dot(output_err, self.weights.T)
        weights_err = np.dot(self.input.T, output_err)

        # Update the parameters:
        self.weights -= alpha * weights_err
        self.bias -= alpha * output_err

        return input_err
