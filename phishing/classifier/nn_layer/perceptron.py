"""Perceptron Layer Module

This module implements a fully-connected perceptron layer for use with the
`NeuralNetwork` class.
"""

import numpy as np

from .base import Layer
from ..nn_utils.activation import SIGMOID


class Perceptron(Layer):
    def __init__(
        self, inputs, outputs, *, activation=SIGMOID
    ) -> None:
        super().__init__(inputs, outputs)

        self.reset()
        self.activation = activation["activation"]
        self.dActivation = activation["derivative"]

        return

    def forward(self, input):
        self.input = input
        # In order to allow the input to come from a typical 2-dimensional
        # matrix (as Pandas will create from reading a CSV file), we check to
        # see if the sample is a vector and if so make it a 1xN matrix. Without
        # this, the backward() function would have a problem with it.
        if len(self.input.shape) == 1:
            self.input = np.array([self.input])
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

    def reset(self):
        self.weights = np.random.rand(self.inputs, self.outputs) - 0.5
        self.bias = np.random.rand(1, self.outputs) - 0.5
