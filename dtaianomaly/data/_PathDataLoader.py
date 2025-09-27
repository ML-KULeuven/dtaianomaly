import abc
import os
from pathlib import Path

from dtaianomaly.data._LazyDataLoader import LazyDataLoader
from dtaianomaly.type_validation import PathAttribute

__all__ = ["PathDataLoader", "from_directory"]


class PathDataLoader(LazyDataLoader, abc.ABC):
    """
    Data loader from disk.

    A dataloader which reads data from a given path. The data loader
    will load the data that is stored at that path.

    Parameters
    ----------
    path : str
        The path at which the data set is located.
    do_caching : bool, default=False
        Whether to cache the loaded data or not.

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
    path: str | Path, base_type: type[PathDataLoader], **kwargs
) -> list[PathDataLoader]:
    """
    Load all PathDataLoaders in a given path.

    Construct a `PathDataLoader` instance for every file in the given `directory`

    Parameters
    ----------
    path : str or Path
        Path to the directory in question.
    base_type : PathDataLoader **object**
        Class object of the data loader, called for constructing
        each data loader instance.
    **kwargs
        Additional arguments to be passed to the dataloader.

    Returns
    -------
    List[PathDataLoader]
        A list of the initialized data loaders, one for each data set in the
        given directory.

    Raises
    ------
    FileNotFoundError
        If `directory` cannot be found
    """
    if not Path(path).is_dir():
        raise ValueError(f"No such directory: {path}")

    all_files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) or os.path.isdir(os.path.join(path, f))
    ]
    return [base_type(file, **kwargs) for file in all_files]
