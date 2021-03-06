"""Base Layer Module

This module provides the base class for defining neural network layers. It is
mainly to allow type-testing of layers passed to the `add` method of the
`NeuralNetwork` class.
"""


class Layer():
    def __init__(self, inputs, outputs) -> None:
        # Each layer needs to hold on to its input and output values during the
        # forward propagation, because they're needed in the calculations of
        # the backward propagation.
        self.input = None
        self.output = None
        # Also store the input/output sizes, so that the `NeuralNetwork` can
        # sanity-check the stacking of layers.
        self.inputs = inputs
        self.outputs = outputs

        return

    # Abstract method for computing the output Y of a layer for an input X.
    # This will raise an exception unless the implementing class provides their
    # own version.
    def forward(self, input):
        raise NotImplementedError

    # Abstract method for computing δE/δX for a given δE/δY. This will raise an
    # exception unless the implementing class provides their own version.
    def backward(self, output_err, alpha):
        raise NotImplementedError

    # Abstract method for resetting a layer as part of fitting a new model to
    # the containing neural network. This will raise an exception unless the
    # implementing class provides their own version.
    def reset(self):
        raise NotImplementedError
