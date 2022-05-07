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
    x = sigmoid(x)
    return x * (1 - x)


# Sigmoid classification
def c_sigmoid(x):
    out = [0 if xx <= 0.5 else 1 for xx in x.ravel()]
    return np.array(out).reshape(x.shape)


# tanh
def tanh(x):
    return np.tanh(x)


# δ(tanh)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2


# Tanh classification
def c_tanh(x):
    out = [0 if xx <= 0.0 else 1 for xx in x.ravel()]
    return np.array(out).reshape(x.shape)


# These dictionary objects group together an activation function, it's
# derivative, and the related classification function. These are what should
# be used to set up layers and configure the classification function on the
# network itself.
SIGMOID = {
    "activation": sigmoid,
    "derivative": d_sigmoid,
    "classification": c_sigmoid,
}
TANH = {
    "activation": tanh,
    "derivative": d_tanh,
    "classification": c_tanh,
}
