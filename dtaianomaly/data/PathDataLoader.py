import abc
import os
from pathlib import Path
from typing import List, Type, Union

from dtaianomaly.data.LazyDataLoader import LazyDataLoader


class PathDataLoader(LazyDataLoader, abc.ABC):
    """
    A dataloader which reads data from a given path. The data loader
    will load the data that is stored at that path.

    Parameters
    ----------
    path: str
        The path at which the data set is located.

    Raises
    ------
    FileNotFoundError
        If the given path does not point to an existing file or directory.
    """

    path: str

    def __init__(self, path: Union[str, Path], do_caching: bool = False):
        super().__init__(do_caching)
        if not (Path(path).is_file() or Path(path).is_dir()):
            raise FileNotFoundError(f"No such file or directory: {path}")
        self.path = str(path)


def from_directory(
    directory: Union[str, Path], dataloader: Type[PathDataLoader], **kwargs
) -> List[PathDataLoader]:
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
        raise FileNotFoundError(f"No such directory: {directory}")

    all_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        or os.path.isdir(os.path.join(directory, f))
    ]
    return [dataloader(file, **kwargs) for file in all_files]
