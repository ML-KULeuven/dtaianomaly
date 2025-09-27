import numpy as np

from dtaianomaly.data._DataSet import DataSet
from dtaianomaly.data._PathDataLoader import PathDataLoader

__all__ = ["UCRLoader"]


class UCRLoader(PathDataLoader):
    """
    Lazy dataloader for the UCR suite of anomaly detection data sets :cite:`wu2023current`.

    The UCR time series anomaly archive consists of 250 time series, which have been published
    to mitigate common issues in existing time series anomaly detection benchmarks:
    (1) Triviality: many benchmarks are easily solved without any fancy algorithms;
    (2) Unrealistic anomaly density: the number of ground truth anomalies is relatively high, even though anomalies should be rare observations;
    (3) Mislabeling: the ground truth labels might not be perfectly aligned with the actual anomalies in the data;
    (4) Run-to-failure bias: most anomalies are located near the end of the time series.

    Parameters
    ----------
    path : str
        The path at which the data set is located.
    do_caching : bool, default=False
        Whether to cache the loaded data or not.

    Notes
    -----
    This implementation expects the file names to contain the start and
    stop time stamps of the single anomaly in the time series as:
    ``*_<train-test-split>_<start>_<stop>.txt``.

    Examples
    --------
    >>> from dtaianomaly.data import UCRLoader
    >>> path_to_ucr = "001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt"
    >>> ucr_data_set = UCRLoader(path_to_ucr).load()  # doctest: +SKIP
    """

    def _load(self) -> DataSet:

        # Extract the meta-information from the name of the file
        [*_, train_test_split, start_anomaly, end_anomaly] = self.path.rstrip(
            ".txt"
        ).split("_")
        train_test_split = int(train_test_split)
        start_anomaly = int(start_anomaly)
        end_anomaly = int(end_anomaly)

        # Load time series
        X = np.loadtxt(self.path)
        X_train = X[:train_test_split]
        X_test = X[train_test_split:]

        # To ensure the file extensions gets ignored
        y = np.zeros(shape=X.shape[0], dtype=int)
        y[start_anomaly:end_anomaly] = 1
        y_test = y[train_test_split:]

        # Return a DataSet object
        return DataSet(X_test=X_test, y_test=y_test, X_train=X_train)
