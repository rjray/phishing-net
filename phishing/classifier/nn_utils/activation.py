"""Activation Functions Module

This module collects the activation functions and their derivatives, that can
be used with the various `Layer`-derived classes.
"""

import numpy as np


# Sigmoid (σ)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# δ(σ)
def d_sigmoid(x):
    predict = sigmoid(x)
    return predict * (1 - predict)


# tanh
def tanh(x):
    return np.tanh(x)


# δ(tanh)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2
