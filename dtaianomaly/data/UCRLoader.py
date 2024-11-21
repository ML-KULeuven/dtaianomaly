
import numpy as np
from dtaianomaly.data.DataSet import DataSet
from dtaianomaly.data.LazyDataLoader import LazyDataLoader


class UCRLoader(LazyDataLoader):
    """
    Lazy dataloader for the UCR suite of anomaly detection data sets.

    This implementation expects the file names to contain the start and
    stop time stamps of the single anomaly in the time series as:
    ``*_<train-test-split>_<start>_<stop>.txt``.
    """

    def _load(self) -> DataSet:

        # Extract the meta-information from the name of the file
        [*_, train_test_split, start_anomaly, end_anomaly] = self.path.rstrip('.txt').split('_')
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
