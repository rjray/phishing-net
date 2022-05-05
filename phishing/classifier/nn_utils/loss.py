"""Loss-Calculation Functions Module

This module collects the loss functions and the derivative functions that can
be used with the `NeuralNetwork` class.
"""

import numpy as np


# Mean Squared Error loss function.
def mse(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))


# Derivative of the MSE function.
def d_mse(y, y_hat):
    return 2 * (y_hat - y) / y.size
