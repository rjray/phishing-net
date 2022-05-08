#!/usr/bin/env python

import os
import sys

_root_dir = os.path.dirname(__file__)
sys.path.append(_root_dir)

import argparse
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from phishing.classifier.logistic import LogisticRegression
from phishing.classifier.neural_net import NeuralNetwork
from phishing.classifier.nn_layer.perceptron import Perceptron
from phishing.data.dataset import Dataset


def parse_command_line():
    parser = argparse.ArgumentParser()

    # Set up the arguments:
    parser.add_argument(
        "-S",
        "--seed",
        type=int,
        default=None,
        help="Specify the random seed to be used for all dataset splitting"
    )
    parser.add_argument(
        "-a",
        "--alphas",
        type=str,
        default="0.1,0.2,0.3",
        help="Comma-separated list of α values to run"
    )
    parser.add_argument(
        "-p",
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory in which to write plots"
    )
    parser.add_argument(
        "-s",
        "--stats-dir",
        type=str,
        default="plots",
        help="Directory in which to write statistics data"
    )

    return vars(parser.parse_args())


# Calculate the determinization of of the model's predictions against the given
# threshold.
def determinize(thresh, positives):
    return np.array([1 if p >= thresh else 0 for p in positives])


def determinization_points(labels, positives, N=1000):
    # Create an array of N+1 points with the determinization values for the
    # threshold probability at each.
    points = []

    for t in range(N + 1):
        thresh = t / N
        det = determinize(thresh, positives)
        mat = confusion_matrix(labels, det)
        tpr = mat[1][1] / (mat[1][1] + mat[1][0])
        fpr = mat[0][1] / (mat[0][1] + mat[0][0])

        points.append((thresh, fpr, tpr))

    return points


def run_one_model_data_combo(type, datatype, instance, ds, alphas):
    # Unlike the homework assignments, here we aren't fishing around for ideal
    # parameters before running the test evaluation. Rather, we will
    # immediately fit the model to the combined train/validation data and base
    # the stats and the plots on the test data.

    print(f"Processing {type} model with {datatype} dataset.")

    X_base = ds.X_base
    y_base = ds.y_base
    X_test = ds.X_test
    y_test = ds.y_test

    results = []

    for alpha in alphas:
        stats_data = {
            "alpha": alpha,
            "type": type,
            "dataset": datatype
        }

        print(f"  α={alpha}")
        print("    Fitting base data")
        model = instance.fit(X_base, y_base, alpha)
        print("    Getting probabilities and prediction")
        probabilities = model.predict_proba(X_test)
        prediction = model.predict(X_test)
        print("    Doing determinization for ROC curve")
        points = determinization_points(y_test, probabilities)
        curve_pts = sorted([[x, y] for _, x, y in points], key=itemgetter(0))
        print("    Calculating best Youden Index")
        youden = sorted(
            [((tpr - fpr), t) for t, fpr, tpr in points],
            key=itemgetter(0), reverse=True
        )[0]

        stats_data["auc"] = roc_auc_score(y_test, probabilities)
        stats_data["risk"] = np.sum(np.abs(prediction - y_test)) / y_test.size
        stats_data["score"] = 1 - stats_data["risk"]
        stats_data["youden"] = youden
        stats_data["curve_pts"] = curve_pts

        results.append(stats_data)

    return results


def main():
    args = parse_command_line()

    ds_labels = ["all", "no_3rd", "url_only"]
    # Set up the datasets for logistic regression (need the bias column):
    ds_bias = [
        Dataset(),
        Dataset(exclude="third_party"),
        Dataset(exclude=["third_party", "content"])
    ]
    # Set up the datasets for the others (don't need the bias column):
    ds = [
        Dataset(bias=False),
        Dataset(bias=False, exclude="third_party"),
        Dataset(bias=False, exclude=["third_party", "content"])
    ]


if __name__ == "__main__":
    main()
