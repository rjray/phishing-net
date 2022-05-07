"""Logistic Regression Module

This module implements logistic regression using stochastic gradient descent,
and is informed by this web resource:

https://machinelearningmastery.com/logistic-regression-tutorial-for-machine-learning/
"""

import numpy as np

MAX_ITERS = 1000
"""Maximum number of iterations to be done when testing by ε."""


class LogisticRegression():
    """The `LogisticRegression` class is a basic implementation of the
    logistic regression algorithm for classification. It expects the feature
    matrices passed to the `fit` and other methods to have the bias/intercept
    column already added."""

    def __init__(self) -> None:
        """Constructor for the Logistic Regression implementation class. Sets
        up slots for the later use by the `fit` method."""

        self.current_weights = None
        self.iterations = None
        self.alpha = None
        self.epsilon = None
        self.audit_trail = None

    def fit(self, X, y, *, alpha=0.1, iterations=MAX_ITERS, epsilon=None):
        f"""Fit a model using the logistic regression algorithm. Takes the
        feature matrix and target vector and derives the weights that can then
        be used for later predictions.

        Positional parameters:

            `X`: The feature matrix, usually a Numpy or Pandas matrix instance.
            `y`: The target vector corresponding to the values of the features.
            in `X`

        Keyword parameters:

            `alpha`: The learning rate to use in the gradient descent
            calculations, defaults to 0.1.
            `iterations`: The number of iterations to run, rather than
            iterating until the mean square of deltas falls below a given ε
            value. Defaults to {MAX_ITERS}.
            `epsilon`: If given, a value to use in deciding when to stop
            iterating the algorithm. Defaults to `None`.

        Note that even when using the `epsilon` value, the algorithm will only
        iterate a maximum of {MAX_ITERS} iterations.

        Returns the calling object, after setting the following attributes on
        the object:

            `current_weights`
            `iterations`
            `alpha`
            `epsilon`
            `audit_trail`:
        """

        # Initialize the weights to all zeros
        weights = np.zeros(X.shape[1])
        # If they passed a value for ε, set this Boolean so we compare against
        # it instead of just a fixed number of iterations.
        stop_by_epsilon = True if epsilon is not None else False
        # Counter for the iterations.
        iteration = 0
        # Keep an audit trail of the MSE rate.
        trail = []

        while True:
            if stop_by_epsilon and iteration == MAX_ITERS:
                print(
                    f"fit: Failed to reach epsilon value within {MAX_ITERS} "
                    "iterations"
                )
                break

            predict = 1. / (1. + np.exp(-X.dot(weights)))
            deltas = X.T.dot((y - predict) * predict * (1 - predict))
            mse = np.mean(np.power(y - predict, 2))
            trail.append(mse)
            weights += alpha * deltas

            if stop_by_epsilon:
                if mse < epsilon:
                    break
            else:
                if iteration == iterations:
                    break

            iteration += 1

        # Save all the relevant parameters from this fitting:
        self.current_weights = weights
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

            `X`: The feature matrix, usually a Numpy or Pandas matrix instance.
        """

        return 1. / (1. + np.exp(-X.dot(self.current_weights)))

    def predict(self, X):
        """Calculate a prediction for the given feature matrix `X`. Returns
        the vector of predictions with the same dimension as the first
        dimension of `X`. Each value will be one of `0` or `1`, based on the
        predicted probability.

        Positional parameters:

            `X`: The feature matrix, usually a Numpy or Pandas matrix instance.
        """

        return np.array([0 if x < 0.5 else 1 for x in self.predict_proba(X)])
