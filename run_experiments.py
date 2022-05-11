#!/usr/bin/env python

import os
import sys

_root_dir = os.path.dirname(__file__)
sys.path.append(_root_dir)

import argparse
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from timeit import default_timer as timer

from phishing.classifier.logistic import LogisticRegression
from phishing.classifier.neural_net import NeuralNetwork
from phishing.classifier.nn_layer.perceptron import Perceptron
from phishing.data.dataset import Dataset


descriptions = {
    "all": "full dataset",
    "no_3rd": "no third-party tool data",
    "url_only": "URL-derived data only",
}


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
        "--stats-file",
        type=str,
        default="data.csv",
        help="File in which to write statistics data"
    )

    return vars(parser.parse_args())


def elapsed(start, end):
    e = end - start
    e_int = int(e)
    mins = int(e_int / 60)
    secs = e - (mins * 60)

    if mins > 0:
        elapsed = f"{mins}m{secs:.3f}s"
    else:
        elapsed = f"{secs:.3f}s"

    return elapsed


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


def run_one_model_data_combo(type, datatype, instance, ds, alpha):
    # Unlike the homework assignments, here we aren't fishing around for ideal
    # parameters before running the test evaluation. Rather, we will
    # immediately fit the model to the combined train/validation data and base
    # the stats and the plots on the test data.

    X_base = ds.X_base
    y_base = ds.y_base
    X_test = ds.X_test
    y_test = ds.y_test

    stats_data = {
        "alpha": alpha,
        "type": type,
        "dataset": datatype
    }

    indent = "    "
    if type.startswith("Neural"):
        indent += "  "

    print(f"{indent}α={alpha}")
    print(f"{indent}  Fitting base data")
    model = instance.fit(X_base, y_base, alpha=alpha)
    print(f"{indent}  Getting probabilities and prediction")
    probabilities = model.predict_proba(X_test)
    prediction = model.predict(X_test)
    print(f"{indent}  Doing determinization for ROC curve")
    points = determinization_points(y_test, probabilities)
    curve_pts = sorted([[x, y] for _, x, y in points], key=itemgetter(0))
    print(f"{indent}  Calculating best Youden Index")
    youden = sorted(
        [((tpr - fpr), t) for t, fpr, tpr in points],
        key=itemgetter(0), reverse=True
    )[0]

    # The NeuralNetwork class will return a Mx1-shaped column vector. If
    # we have that, reshape it to a row vector of dimension M.
    if len(prediction.shape) == 2 and prediction.shape[1] == 1:
        prediction = prediction.ravel()

    stats_data["auc"] = roc_auc_score(y_test, probabilities)
    stats_data["risk"] = np.sum(np.abs(prediction - y_test)) / y_test.size
    stats_data["score"] = 1 - stats_data["risk"]
    stats_data["youden"] = youden
    stats_data["curve_pts"] = curve_pts

    return stats_data


def plot_curve(ax, points, label, color):
    pts = np.array(points)
    ax.plot(pts[:, 0], pts[:, 1], lw=1, color=color, label=label)

    return


def generate_plots(data, title, filename):
    # Colors we'll use:
    colors = ["red", "green", "blue"]

    # Set up the 2x2 grid of plots.
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(title)

    for idx, run in enumerate(data):
        i = 1 if idx & 2 else 0
        j = 1 if idx & 1 else 0
        label = f"α={run['alpha']}"
        plot_curve(axs[i][j], run["curve_pts"], label, colors[idx])
        plot_curve(axs[1][1], run["curve_pts"], label, colors[idx])

    for i in range(2):
        for j in range(2):
            axs[i][j].plot(
                [0, 1], [0, 1], color="navy", lw=1, linestyle="--", marker="."
            )
            axs[i][j].set_xlim([-0.05, 1.05])
            axs[i][j].set_ylim([-0.05, 1.05])
            axs[i][j].grid()
            axs[i][j].legend(loc="lower right")

    fig.savefig(filename)

    return


def main():
    args = parse_command_line()
    seed = args["seed"]
    alphas = list(map(float, args["alphas"].split(",")))

    # Set up the datasets:

    ds_labels = ["all", "no_3rd", "url_only"]
    # Set up the datasets for logistic regression (need the bias column):
    ds_bias = [
        Dataset(bias=True),
        Dataset(bias=True, exclude="third_party"),
        Dataset(bias=True, exclude=["third_party", "content"])
    ]
    for d in ds_bias:
        d.create_split(0.2, random_test=seed, random_validate=seed)
    # Set up the datasets for the others (don't need the bias column):
    ds = [
        Dataset(),
        Dataset(exclude="third_party"),
        Dataset(exclude=["third_party", "content"])
    ]
    for d in ds:
        d.create_split(0.2, random_test=seed, random_validate=seed)

    # Gather all the data from all the runs:
    data = []

    # Start gathering data.
    run_started = timer()

    # Obtain data for the logistic regression models:
    print("Processing LogisticRegression classifier")
    start = timer()
    for ds_index in range(3):
        print(f"  Using dataset {ds_labels[ds_index]}")
        row = []
        ds_start = timer()
        for alpha in alphas:
            row.append(
                run_one_model_data_combo(
                    "LogisticRegression",
                    ds_labels[ds_index],
                    LogisticRegression(),
                    ds_bias[ds_index],
                    alpha
                )
            )
        ds_end = timer()
        data.append(row)
        print(f"  Dataset completed ({elapsed(ds_start, ds_end)})")
    end = timer()
    print(f"LogisticRegression processing complete ({elapsed(start, end)})")

    # Now gather for the neural network models. Note that each dataset calls
    # for slightly different NN instances.
    print("Processing NeuralNetwork classifier")
    start = timer()
    for ds_index in range(3):
        print(f"  Using dataset {ds_labels[ds_index]}")
        ds_start = timer()

        dataset = ds[ds_index]
        # Each of the three datasets is a different size
        features = dataset.X.shape[1]
        base2 = math.ceil(math.log2(features))
        nn_list = []
        labels = []
        # We're going to run three different NNs over each dataset:
        #
        #   1. No hidden layer
        #   2. One hidden layer of 2*base2 size
        #   3. Two hidden layers of base2 size
        labels.append(f"NeuralNetwork({features},1)")
        nn_list.append(
            NeuralNetwork(
                Perceptron(features, 1)
            )
        )
        labels.append(f"NeuralNetwork({features},{base2 * 2},1)")
        nn_list.append(
            NeuralNetwork(
                Perceptron(features, base2 * 2),
                Perceptron(base2 * 2, 1)
            )
        )
        labels.append(f"NeuralNetwork({features},{base2},{base2},1)")
        nn_list.append(
            NeuralNetwork(
                Perceptron(features, base2),
                Perceptron(base2, base2),
                Perceptron(base2, 1)
            )
        )

        for nn_index in range(3):
            print(f"    Using {labels[nn_index]}")
            row = []
            for alpha in alphas:
                row.append(
                    run_one_model_data_combo(
                        labels[nn_index],
                        ds_labels[ds_index],
                        nn_list[nn_index],
                        dataset,
                        alpha
                    )
                )
                nn_list[nn_index].reset()
            print(f"    {labels[nn_index]} complete")
            data.append(row)

        ds_end = timer()
        print(f"  Dataset completed ({elapsed(ds_start, ds_end)})")

    end = timer()
    print(f"NeuralNetwork processing complete ({elapsed(start, end)})")

    run_finished = timer()
    print(f"\nTotal data-time: {elapsed(run_started, run_finished)}")

    print(f"\nWriting {args['stats_file']}")
    with open(args["stats_file"], "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            ["method", "dataset", "learning_rate", "auc", "score", "youden",
             "threshold"]
        )
        for row in data:
            for inst in row:
                youden = inst["youden"]
                writer.writerow(
                    [inst["type"], inst["dataset"], inst["alpha"], inst["auc"],
                     inst["score"], youden[0], youden[1]]
                )

    print("\nCreating plots")
    if not os.path.exists(args["plots_dir"]):
        os.makedirs(args["plots_dir"])

    for row in data:
        type = row[0]["type"]
        dataset = row[0]["dataset"]
        ds_desc = descriptions[dataset]

        title = f"{type} on {ds_desc}"
        filename = f"{type.lower()}-{dataset}.png"
        filename = os.path.join(args["plots_dir"], filename)
        print(f"  writing {filename}")
        generate_plots(row, title, filename)

    print("\nRunning comparison algorithms:")
    print("\n  BayesianRidge")
    for ds_index in range(3):
        print(f"    Using dataset {ds_labels[ds_index]}:")
        dataset = ds[ds_index]
        X_base = dataset.X_base
        y_base = dataset.y_base
        X_test = dataset.X_test
        y_test = dataset.y_test
        clf = BayesianRidge().fit(X_base, y_base)
        prediction = clf.predict(X_test)
        score = 1 - (np.sum(np.abs(prediction - y_test)) / y_test.size)
        cod = clf.score(X_test, y_test)
        print(f"      score={score}")
        print(f"      coefficient of determination={cod}")

    print("\n  DecisionTreeClassifier")
    for ds_index in range(3):
        print(f"    Using dataset {ds_labels[ds_index]}:")
        dataset = ds[ds_index]
        X_base = dataset.X_base
        y_base = dataset.y_base
        X_test = dataset.X_test
        y_test = dataset.y_test
        clf = DecisionTreeClassifier().fit(X_base, y_base)
        prediction = clf.predict(X_test)
        score = 1 - (np.sum(np.abs(prediction - y_test)) / y_test.size)
        ma = clf.score(X_test, y_test)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        print(f"      score={score}")
        print(f"      mean accuracy={ma}")
        print(f"      AUC={auc}")

    print("\nComplete.")

    return


if __name__ == "__main__":
    main()
