"""Scoring Module

Provide the `score()` and `risk()` functions for evaluating models.
"""

import numpy as np


def risk(model, X, y):
    """Return the calculated empirical risk of the fitted model by making
    a prediction for the feature matrix `X` and comparing it to the target
    vector `y`.

    Positional parameters:

        `X`: The feature matrix, usually a Numpy or Pandas matrix instance
        `y`: The target vector corresponding to the true values of the
        features in `X`
    """

    prediction = model.predict(X)
    return np.sum(np.abs(prediction - y)) / len(y)


def score(model, X, y):
    """Return the accuracy score of the fitted model by making a prediction
    for the feature matrix `X` and comparing it to the target vector `y`.

    Positional parameters:

        `X`: The feature matrix, usually a Numpy or Pandas matrix instance
        `y`: The target vector corresponding to the true values of the
        features in `X`
    """

    return 1 - model.risk(X, y)
