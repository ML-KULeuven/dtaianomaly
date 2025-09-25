import abc
import os
from pathlib import Path

from dtaianomaly.data._LazyDataLoader import LazyDataLoader
from dtaianomaly.type_validation import PathAttribute

__all__ = ["PathDataLoader", "from_directory"]


class PathDataLoader(LazyDataLoader, abc.ABC):
    """
    A dataloader which reads data from a given path. The data loader
    will load the data that is stored at that path.

    Parameters
    ----------
    path: str
        The path at which the data set is located.
    do_caching: bool, default=False
        Whether to cache the loaded data or not

    Raises
    ------
    FileNotFoundError
        If the given path does not point to an existing file or directory.
    """

    path: str | Path

    attribute_validation = {"path": PathAttribute()}

    def __init__(self, path: str | Path, do_caching: bool = False):
        super().__init__(do_caching)
        self.path = str(path)


def from_directory(
    directory: str | Path, dataloader: type[PathDataLoader], **kwargs
) -> list[PathDataLoader]:
    """
    Construct a `PathDataLoader` instance for every file in the given `directory`

    Parameters
    ----------
    directory: str or Path
        Path to the directory in question
    dataloader: PathDataLoader **object**
        Class object of the data loader, called for constructing
        each data loader instance
    **kwargs:
        Additional arguments to be passed to the dataloader

    Returns
    -------
    data_loaders: List[PathDataLoader]
        A list of the initialized data loaders, one for each data set in the
        given directory.

    Raises
    ------
    FileNotFoundError
        If `directory` cannot be found
    """
    if not Path(directory).is_dir():
        raise ValueError(f"No such directory: {directory}")

    all_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        or os.path.isdir(os.path.join(directory, f))
    ]
    return [dataloader(file, **kwargs) for file in all_files]
