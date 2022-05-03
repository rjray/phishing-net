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

# The following lists define the subsets of features from the datasets. These
# will be further combined into sets that can be selectively dropped from the
# overall dataset. This includes some combinations and subsets that likely will
# not be used in the long run.

URL_STRUCT_FEATURES = [
    "ip",
    "https_token",
    "punycode",
    "port",
    "tld_in_path",
    "tld_in_subdomain",
    "abnormal_subdomain",
    "prefix_suffix",
    "random_domain",
    "shortening_service",
    "path_extension",
    "domain_in_brand",
    "brand_in_subdomain",
    "brand_in_path",
    "suspecious_tld",
    "statistical_report",
]
URL_STAT_FEATURES = [
    "length_url",
    "length_hostname",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_or",
    "nb_eq",
    "nb_underscore",
    "nb_tilde",
    "nb_percent",
    "nb_slash",
    "nb_star",
    "nb_colon",
    "nb_comma",
    "nb_semicolumn",
    "nb_dollar",
    "nb_space",
    "nb_www",
    "nb_com",
    "nb_dslash",
    "http_in_path",
    "ratio_digits_url",
    "ratio_digits_host",
    "nb_subdomains",
    "nb_redirection",
    "nb_external_redirection",
    "length_words_raw",
    "char_repeat",
    "shortest_words_raw",
    "shortest_word_host",
    "shortest_word_path",
    "longest_words_raw",
    "longest_word_host",
    "longest_word_path",
    "avg_words_raw",
    "avg_word_host",
    "avg_word_path",
    "phish_hints",
]

CONTENT_HYPERLINKS_FEATURES = [
    "nb_hyperlinks",
    "ratio_intHyperlinks",
    "ratio_extHyperlinks",
    "ratio_nullHyperlinks",
    "nb_extCSS",
    "ratio_intRedirection",
    "ratio_extRedirection",
    "ratio_intErrors",
    "ratio_extErrors",
    "external_favicon",
    "links_in_tags",
    "ratio_intMedia",
    "ratio_extMedia",
]
CONTENT_ABNORMALNESS_FEATURES = [
    "login_form",
    "submit_email",
    "sfh",
    "iframe",
    "popup_window",
    "safe_anchor",
    "onmouseover",
    "right_clic",
    "empty_title",
    "domain_in_title",
    "domain_with_copyright",
]

THIRD_PARTY_FEATURES = [
    "whois_registered_domain",
    "domain_registration_length",
    "domain_age",
    "web_traffic",
    "dns_record",
    "google_index",
    "page_rank",
]

FEATURES_MAP = {
    "url": URL_STRUCT_FEATURES + URL_STAT_FEATURES,
    "url_struct": URL_STRUCT_FEATURES,
    "url_stat": URL_STAT_FEATURES,
    "content": CONTENT_HYPERLINKS_FEATURES + CONTENT_ABNORMALNESS_FEATURES,
    "content_hyperlinks": CONTENT_HYPERLINKS_FEATURES,
    "content_abnormalness": CONTENT_ABNORMALNESS_FEATURES,
    "third_party": THIRD_PARTY_FEATURES,
}
"""Features of the dataset broken down into groups for potential exclusion."""

CACHE_MAP = dict()
"""A dict/map for caching the CSV file(s) read in."""


class Dataset():
    f"""The Dataset class encapsulates the dataset being used for this project.
    An instance created of this class will (by default) read the data from the
    file:

    {DATASET}

    However, the path to the file can be passed as an argument to the
    constructor.
    """

    def __init__(self, file=DATASET, *, exclude=None, nobias=False) -> None:
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

            `exclude`: If passed and not `None`, then it indicates subsets of
            features that should be dropped from the dataset after it is read
            but before it is returned as an object. The value may be a string
            or a list of strings.
            `nobias`: If passed and not `False`, then the bias column will not
            be added to the constructed feature matrix. This is useful for
            using the dataset with SciKit classes which do not need the
            explicit bias column present.
        """

        # First get the data itself, either from the cache or from file:
        if file in CACHE_MAP:
            df = CACHE_MAP[file]
        else:
            df = pd.read_csv(file)
            CACHE_MAP[file] = df

        # Always drop these two from the feature matrix:
        self.X = df.drop(["url", "status"], axis=1)
        self.y = df["status"].transform(
            lambda v: 1 if v == "phishing" else 0
        )

        # If one or more subsets of the features were requested for exclusion,
        # process that here.
        if exclude:
            if isinstance(exclude, str):
                self.X.drop(FEATURES_MAP[exclude], axis=1, inplace=True)
            else:
                for fea_set in exclude:
                    self.X.drop(FEATURES_MAP[fea_set], axis=1, inplace=True)

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

        # Note the file used for this object:
        self.file = file

        return

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
