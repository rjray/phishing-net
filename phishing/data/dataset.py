"""Dataset Module

This module encapsulates the dataset used for this project. It reads the data
for each instance of the class created, and provides a method for creating the
train/validate/test split of the data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_this_dir = os.path.dirname(__file__)
DATASET = os.path.abspath(f"{_this_dir}/../../data/dataset_phishing.csv")
"""The default dataset file. See references."""
UNUSED_FEATURES = [
    "whois_registered_domain",
    "domain_registration_length",
    "domain_age",
    "web_traffic",
    "dns_record",
    "google_index",
    "page_rank",
]
"""Features of the dataset not used in the regression by default."""


class Dataset():
    f"""The Dataset class encapsulates the dataset being used for this project.
    An instance created of this class will (by default) read the data from the
    file:

    {DATASET}

    However, the path to the file can be passed as an argument to the
    constructor.
    """

    def __init__(self, file=DATASET, *, all=False, nobias=False) -> None:
        f"""The constructor for this class, called automatically when an
        instance of the class is created. This function reads the dataset as a
        CSV file into a Pandas dataframe object, storing the whole dataframe on
        the object as well as creating `X` and `y` attributes which represent
        the data and the truth values respectively. It also sets up empty
        attributes for the X/y slices of the data for training, validation and
        testing.

        The constructor takes one positional argument:

            `file`: The name of the file to read the data from. Defaults to
            {DATASET}

        The constructor also recognizes two optional keyword arguments:

            `all`: If passed and not `False`, then all features from the
            dataset are used. If not passed (or passed as `False`), then some
            features are dropped and not considered for classification.
            `nobias`: If passed and not `False`, then the bias column will not
            be added to the constructed feature matrix. This is useful for
            using the dataset with SciKit classes which do not need the
            explicit bias column present.
        """
        self.file = file
        self.df = pd.read_csv(file)
        # Always drop these two from the feature matrix:
        self.X = self.df.drop(["url", "status"], axis=1)
        # Unless "all" is true, also drop these:
        if not all:
            self.X.drop(UNUSED_FEATURES, axis=1, inplace=True)
        self.y = self.df["status"].transform(
            lambda v: 1 if v == "phishing" else 0
        )

        # Normalize the features of X with Min-Max Normalization
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
        # That may leave some NaN values in self.X, replace them with 0.
        self.X = self.X.fillna(0)

        # Add the bias column to X unless "nobias" is set to True.
        if not nobias:
            bias = pd.DataFrame(np.ones(self.X.shape[0]), columns=["bias"])
            self.X = pd.concat([bias, self.X], axis=1)

        # Create placeholders for the split buckets.
        self.X_base = None
        self.X_train = None
        self.X_validate = None
        self.X_test = None
        self.y_base = None
        self.y_train = None
        self.y_validate = None
        self.y_test = None

    def create_split(
        self, test, validate=None, *, random_test=None, random_validate=None
    ) -> None:
        """Creates the split between training, validation and testing segments
        of the dataset.

        This method takes two positional arguments:

            `test`: The percentage of the data that should be sliced out
            (randomly) for testing the model. The value should be a float
            between 0 and 1, non-inclusive.
            `validate`: An optional second parameter to specify what percentage
            of the data should be set aside for validation. This is a share of
            the whole, not of the training split. For example, if `test` and
            `validate` are both set to 0.20 (requesting a 60/20/20 split) then
            the validation slice will actually be 25% of the training slice so
            as to meet the requested percentage. If this parameter is not
            given, it defaults to the value of the `test` parameter.

        This method also recognizes two optional keyword arguments:

            `random_test`: If given, should be an integer that will be used as
            the random seed in the selection of the test slice of data.
            Defaults to `None`, which will cause `train_test_split` to use the
            existing randomness setting.
            `random_validate`: As above, for the selection of the validation
            data slice.

        Upon return, the calling object will have eight attributes filled in:

            `X_train`/`y_train`: The slices of the data that will be used for
            training.
            `X_validate`/`y_validate`: The slices of the data that will be
            used for validation.
            `X_test`/`y_test`: The slices of the data that will be used for
            testing.
            `X_base`/`y_base`: The combined training and validation data, for
            use during the final testing of the model.
        """
        if not validate:
            validate = test / (1 - test)

        self.X_base, self.X_test, self.y_base, self.y_test = train_test_split(
            self.X, self.y, test_size=test, random_state=random_test
        )
        self.X_train, self.X_validate, self.y_train, self.y_validate = \
            train_test_split(
                self.X_base, self.y_base, test_size=validate,
                random_state=random_validate
            )

        return self
