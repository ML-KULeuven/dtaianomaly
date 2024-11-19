
import numpy as np
from typing import NamedTuple


class DataSet(NamedTuple):
    """
    A class for time series anomaly detection data sets. These
    consist of the raw data itself and the ground truth labels.

    Parameters
    ----------
    x: array-like of shape (n_samples, n_features)
        The time series.
    y: array-like of shape (n_samples)
        The ground truth anomaly labels.
    """
    x: np.ndarray
    y: np.ndarray
