import abc

from dtaianomaly.data.DataSet import DataSet
from dtaianomaly.utils import PrintConstructionCallMixin


class LazyDataLoader(PrintConstructionCallMixin):
    """
    A lazy dataloader for anomaly detection workflows

    This is a data loading utility to point towards a specific data set
    and to load it at a later point in time during execution of a workflow.

    This way we limit memory usage and allow for virtually unlimited scaling
    of the number of data sets in a workflow.

    Parameters
    ----------
    do_caching: bool, default=False
        Whether to cache the loaded data or not

    Attributes
    ----------
    cache_ : DataSet
        Cached version of the loaded data set. Only available if ``do_caching==True``
        and the data has been loaded before.
    """

    do_caching: bool
    cache_: DataSet

    def __init__(self, do_caching: bool = False):
        self.do_caching = do_caching

    def load(self) -> DataSet:
        """
        Load the dataset. If ``do_caching==True``, the loaded will be saved in the
        cache if no cache is available yet, and the cached data will be returned.

        Returns
        -------
        data_set: DataSet
            The loaded dataset.
        """
        if self.do_caching:
            if not hasattr(self, "cache_"):
                self.cache_ = self._load()
            return self.cache_
        else:
            return self._load()

    @abc.abstractmethod
    def _load(self) -> DataSet:
        """Abstract method to effectively load the data."""
