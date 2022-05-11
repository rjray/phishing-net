"""Neural Network Module

This module implements a basic neural network of multiple layers, and is
informed/inspired by this web resource:

https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
"""

import numpy as np

from .nn_layer.base import Layer
from .nn_utils.loss import mse, d_mse
from .nn_utils.activation import SIGMOID

MAX_ITERS = 100
"""Maximum number of iterations to be done when training a network."""


class NeuralNetwork():
    """The `NeuralNetwork` class is a basic implementation of a variable-layer
    neural network that uses backpropagation for the learning process. It
    expects the feature matrices to not have bias already added.

    The architecture of this allows for multiple layers to be added to the
    network. See the `Layer` class for the base class used to define layers.
    """

    def __init__(
        self, *layers, lossFn=mse, dLossFn=d_mse, classify=SIGMOID
    ) -> None:
        """Constructor for the Neural Network class. Can optionally take an
        arbitrary list of layers to immediately add to the network. Sets up
        slots for later use by the `fit` method.

        Positional parameters:

            `layers`: Zero or more instances of a layer class (one that derives
            from `Layer`) to be added to the network in the order specified.

        Keyword parameters:

            `lossFn`: The loss-measuring function to use for reporting
            `dLossFn`: The function that implements the derivative of `lossFn`,
            used in calculating error for backpropagation
            `classify`: Either a function to do the classification from the
            set of prediction probabilities, or a structure from the
            `phishing.classifier.nn_utils.activation` module that includes a
            `classification` key to specify the classification function.
        """

        self.layers = []
        self.iterations = None
        self.alpha = None
        self.epsilon = None
        self.audit_trail = None

        self.lossFn = lossFn
        self.dLossFn = dLossFn

        if isinstance(classify, dict):
            self.classify = classify["classification"]
        else:
            self.classify = classify

        if len(layers):
            self.add(*layers)

        return

    def add(self, *layers) -> None:
        """Add one or more layers to the calling neural network object.

        Positional parameters:

            `layers`: Zero or more instances of a layer class (one that derives
            from `phishing.classifier.nn_layer.base.Layer`).
        """

        for ll in layers:
            if not isinstance(ll, Layer):
                raise ValueError(
                    f"Argument {ll} does not derive from the Layer class"
                )

            if len(self.layers) != 0 and ll.inputs != self.layers[-1].outputs:
                layer_in = ll.inputs
                prev_out = self.layers[-1].outputs
                raise ValueError(
                    f"Layer output/input mismatch ({prev_out} != {layer_in})"
                )

            self.layers.append(ll)

    def setLossFns(self, lossFn, dLossFn):
        """Set or change the loss and loss-derivative functions used by the
        instance for reporting and backpropagation."""

        self.lossFn = lossFn
        self.dLossFn = dLossFn

    def fit(self, X, y, *, alpha=0.1, iterations=MAX_ITERS, epsilon=None):
        f"""Fit a model using the backpropagation algorithm. Takes the feature
        matrix and target vector and derives the weights that can then be used
        for later predictions.

        Positional parameters:

            `X`: The feature matrix, usually a Numpy or Pandas matrix instance.
            `y`: The target vector corresponding to the values of the features.
            in `X`

        Keyword parameters:

            `alpha`: The learning rate to use in the gradient descent
            calculations, defaults to 0.1.
            `iterations`: The number of iterations to run, rather than
            iterating until the mean square error falls below a given ε
            value. Defaults to {MAX_ITERS}.
            `epsilon`: If given, a value to use in deciding when to stop
            iterating the algorithm. Defaults to `None`.

        Note that even when using the `epsilon` value, the algorithm will only
        iterate a maximum of {MAX_ITERS} iterations.

        Returns the calling object, after setting the following attributes on
        the object:

            `iterations`
            `alpha`
            `epsilon`
            `audit_trail`:
        """

        # Count the number of samples we're fitting with:
        num_samples = len(X)
        # If they passed a value for ε, set this Boolean so we compare against
        # it instead of just a fixed number of iterations.
        stop_by_epsilon = True if epsilon is not None else False
        # Counter for the iterations.
        iteration = 0
        # Keep an audit trail of the MSD rate.
        trail = []

        # Primary training loop:
        while True:
            if stop_by_epsilon and iteration == MAX_ITERS:
                print(
                    f"fit: Failed to reach epsilon (ε={epsilon}) within "
                    f"{MAX_ITERS} iterations"
                )
                break

            error = 0
            for idx in range(num_samples):
                # Do the forward propagation.
                output = X[idx]
                for layer in self.layers:
                    output = layer.forward(output)

                # Accumulate the loss, for epsilon testing:
                error += self.lossFn(y[idx], output)

                # Now do the backward propagation:
                output_err = self.dLossFn(y[idx], output)
                for layer in reversed(self.layers):
                    output_err = layer.backward(output_err, alpha)

            # Average the forward-error over the number of samples:
            error /= num_samples
            # Save it on the "audit trail":
            trail.append(error)
            # Bump the iteration counter:
            iteration += 1

            if stop_by_epsilon:
                if error < epsilon:
                    break
            else:
                if iteration == iterations:
                    break

        self.iterations = iteration
        self.alpha = alpha
        self.epsilon = epsilon
        self.audit_trail = trail

        return self

    def predict_proba(self, X):
        """Calculate the prediction probabilities for the given feature matrix
        `X`. Returns the vector of probabilities with the same dimension as the
        first dimension of `X`.

        Positional parameters:

            `X`: The feature matrix, usually a Numpy matrix instance.
        """

        # Count the number of samples we're predicting over:
        num_samples = len(X)
        probabilities = []

        # Apply the network to all samples:
        for idx in range(num_samples):
            output = X[idx]

            # For prediction, we just run the forward propagation. No backward
            # propagation in this case.
            for layer in self.layers:
                output = layer.forward(output)

            probabilities.append(output)

        probabilities = np.array(probabilities)
        # If the resulting shape is (N, 1, M), remove the spurious inner 1.
        if len(probabilities.shape) == 3 and probabilities.shape[1] == 1:
            probabilities = probabilities.reshape(
                (probabilities.shape[0], probabilities.shape[2])
            )

        return probabilities

    def predict(self, X):
        """Calculate a prediction for the given feature matrix `X`. Returns
        the vector of predictions with the same dimension as the first
        dimension of `X`. Each value will be one of `0` or `1`, based on the
        predicted probability.

        Positional parameters:

            `X`: The feature matrix, usually a Numpy matrix instance.
        """

        return self.classify(self.predict_proba(X))

    def reset(self):
        """Reset the weights in all layers of the neural network, allowing it
        to be used to fit a new model.
        """

        for layer in self.layers:
            layer.reset()
