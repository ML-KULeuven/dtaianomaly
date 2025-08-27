from pathlib import Path

import numpy as np
import pandas as pd

from dtaianomaly.data.LazyDataLoader import DataSet, LazyDataLoader


class CustomDataLoader(LazyDataLoader):
    """
    A data loader for loading custom data.

    The training and testing data is located in different files. Both must be
    readable through ``pandas.read_csv(path)``. The test data must contain a
    column with name 'label', in which the anomalies are marked (1 for anomaly,
    0 for normal). The test data may have an optional column 'time', which will
    be interpreted as the time step of each observation. All other columns are
    assumed to be part of the time series data. The 'label' column is optional
    for the training set. If note present, the training data is assumed to be
    completely normal. The training data may have an optional column 'time',
    similarly as for the test data. All remaining columns are time series data.
    The titles of the training and test set must match exactly, although the
    order may be different.

    Parameters
    ----------
    test_path: str
        The path at which the test data is located.
    train_path: str, default=None
        The path at which the train data is located. If None, then there
        will be no training data in the loaded dataset.
    do_caching: bool, default=False
        Whether to cache the loaded data or not
    """

    train_path: str | None
    test_path: str

    def __init__(
        self,
        test_path: str | Path,
        train_path: str | Path = None,
        do_caching: bool = False,
    ):
        super().__init__(do_caching)
        for path in [train_path, test_path]:
            if path is not None:
                if not (Path(path).is_file() or Path(path).is_dir()):
                    raise FileNotFoundError(f"No such file or directory: {path}")
        if test_path is None:
            raise FileNotFoundError(f"The test_path may not be None!")

        self.train_path = train_path
        self.test_path = test_path

    def _load(self) -> DataSet:

        # Load test data
        time_steps_test = None
        df_test = pd.read_csv(self.test_path)
        if "time" in df_test.columns:
            time_steps_test = df_test.pop("time").values
        y_test = df_test.pop("label").values
        X_test = df_test.values
        features_test = list(df_test.columns)

        # Load train data
        time_steps_train = None
        y_train = None
        X_train = None
        if self.train_path is not None:
            df_train = pd.read_csv(self.train_path)
            if "time" in df_train.columns:
                time_steps_train = df_train.pop("time").values
            if "label" in df_train.columns:
                y_train = df_train.pop("label").values
                if np.all(y_train == 0):
                    y_train = None

            # Check the features
            features_train = list(df_train.columns)
            if set(features_test) != set(features_train):
                raise ValueError(
                    "The train and test time series consist of different features!"
                    f"Train data has features: {features_train}"
                    f"Test data has features: {features_test}"
                )

            # Make sure the features follow the same order
            X_train = df_train[features_test].values

        return DataSet(
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            y_train=y_train,
            feature_names=features_test,
            time_steps_test=time_steps_test,
            time_steps_train=time_steps_train,
        )
