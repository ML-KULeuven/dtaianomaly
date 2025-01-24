import abc
import os
from pathlib import Path
from typing import List, Type, Union

from dtaianomaly.data.DataSet import DataSet
from dtaianomaly.PrettyPrintable import PrettyPrintable


class LazyDataLoader(PrettyPrintable):
    """
    A lazy dataloader for anomaly detection workflows

    This is a data loading utility to point towards a specific data set
    (with `path`) and to load it at a later point in time during
    execution of a workflow.

    This way we limit memory usage and allow for virtually unlimited scaling
    of the number of data sets in a workflow.

    Parameters
    ----------
    path: str
        Path to the relevant data set.
    do_caching: bool, default=False
        Whether to cache the loaded data or not

    Attributes
    ----------
    cache_ : DataSet
        Cached version of the loaded data set. Only available if ``do_caching==True``
        and the data has been loaded before.

    Raises
    ------
    FileNotFoundError
        If the given path does not point to an existing file or directory.
    """

    path: str
    do_caching: bool
    cache_: DataSet

    def __init__(self, path: Union[str, Path], do_caching: bool = False):
        if not (Path(path).is_file() or Path(path).is_dir()):
            raise FileNotFoundError(f"No such file or directory: {path}")
        self.path = str(path)
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


def from_directory(
    directory: Union[str, Path], dataloader: Type[LazyDataLoader], **kwargs
) -> List[LazyDataLoader]:
    """
    Construct a `LazyDataLoader` instance for every file in the given `directory`

    Parameters
    ----------
    directory: str or Path
        Path to the directory in question
    dataloader: LazyDataLoader **object**
        Class object of the data loader, called for constructing
        each data loader instance
    **kwargs:
        Additional arguments to be passed to the dataloader

    Returns
    -------
    data_loaders: List[LazyDataLoader]
        A list of the initialized data loaders, one for each data set in the
        given directory.

    Raises
    ------
    FileNotFoundError
        If `directory` cannot be found
    """
    if not Path(directory).is_dir():
        raise FileNotFoundError(f"No such directory: {directory}")

    all_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        or os.path.isdir(os.path.join(directory, f))
    ]
    return [dataloader(file, **kwargs) for file in all_files]
