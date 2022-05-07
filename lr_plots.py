#!/usr/bin/env python

import os
import sys

_root_dir = os.path.dirname(__file__)
sys.path.append(_root_dir)

import matplotlib.pyplot as plt

from phishing.data.dataset import Dataset
from phishing.classifier.logistic import LogisticRegression


def main():
    datasets = [
        (Dataset(exclude="third_party").create_split(0.2),
         "regression-regular.png"),
        (Dataset().create_split(0.2), "regression-all.png"),
    ]

    for ds, file in datasets:
        print(f"Creating {file}...")
        _, ax = plt.subplots()

        for alpha in [0.1, 0.2, 0.3]:
            print(f"  Creating LogisticRegression with α={alpha}")
            lr = LogisticRegression().fit(ds.X_train, ds.y_train, alpha=alpha)
            vals = lr.audit_trail
            ax.plot(range(1, len(vals) + 1), vals, label=f"α={alpha}")

        ax.set_xlabel("Gradient Descent Iteration")
        ax.set_ylabel("Mean Squared Error")
        ax.legend()
        plt.savefig(file)
        print(f"{file} written.")


if __name__ == "__main__":
    main()
