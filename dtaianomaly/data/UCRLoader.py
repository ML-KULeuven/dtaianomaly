
import numpy as np
from dtaianomaly.data.DataSet import DataSet
from dtaianomaly.data.LazyDataLoader import LazyDataLoader


class UCRLoader(LazyDataLoader):
    """
    Lazy dataloader for the UCR suite of anomaly detection data sets.

    This implementation expects the file names to contain the start and
    stop time stamps of the single anomaly in the time series as:
    '\*_start_stop.txt'.
    """

    def _load(self) -> DataSet:
        # Load time series
        X = np.loadtxt(self.path)

        # Load anomaly targets (0 is background, 1 is anomaly)
        # UCR datasets specify the anomaly in dataset name
        name = self.path.split('_')

        onset = int(name[-2])
        offset = int(''.join(filter(str.isdigit, name[-1])))
        # To ensure the file extensions gets ignored

        y = np.zeros(shape=X.shape, dtype=np.int8)
        y[onset:offset] = 1

        return DataSet(x=X, y=y)
