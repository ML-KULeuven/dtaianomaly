import os
import pickle
import re
from pathlib import Path

import numpy as np

from dtaianomaly.data._DataSet import DataSet
from dtaianomaly.data._PathDataLoader import PathDataLoader

__all__ = ["ASDLoader"]

_MACHINE_PATTERN = re.compile(r"^omi-\d+$")

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
}


class _NumpyUnpickler(pickle.Unpickler):
    """Unpickler that only reconstructs numpy arrays, to avoid executing arbitrary code."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"Global '{module}.{name}' is not allowed when loading ASD data."
            )
        return super().find_class(module, name)


def _load_pickle(file: str) -> np.ndarray:
    with open(file, "rb") as f:
        return _NumpyUnpickler(f).load()


class ASDLoader(PathDataLoader):
    """
    Lazy dataloader for the Application Server Dataset (ASD) :cite:`li2021multivariate`.

    The Application Server Dataset is collected from a large internet company
    and consists of the data of 12 servers, each with 19 metrics characterizing
    the status of the server. The data of each server is split into a train and
    test set, of which only the test set is labeled.

    Parameters
    ----------
    path : str
        The path to the root of the dataset, which contains the folder ``processed``.
    machine : str
        The name of the server to load, of the form ``omi-<k>``.
    do_caching : bool, default=False
        Whether to cache the loaded data or not.

    Raises
    ------
    ValueError
        If ``machine`` does not have the form ``omi-<k>``.
    FileNotFoundError
        If the train, test or label file of the server does not exist.

    Notes
    -----
    This implementation expects the files of the dataset to be organized as
    ``<path>/processed/<machine>_train.pkl``, ``<path>/processed/<machine>_test.pkl``
    and ``<path>/processed/<machine>_test_label.pkl``, as in the official release.
    Because ``path`` points to the root of the dataset, :py:func:`~dtaianomaly.data.from_directory`
    can not be used with this loader. The data always has two dimensions. For
    safety, the pickle files may only contain numpy arrays.

    Examples
    --------
    >>> from dtaianomaly.data import ASDLoader
    >>> asd_data_set = ASDLoader("ASD", "omi-1").load()  # doctest: +SKIP
    """

    machine: str

    def __init__(self, path: str | Path, machine: str, do_caching: bool = False):
        super().__init__(path, do_caching)
        if not isinstance(machine, str) or _MACHINE_PATTERN.match(machine) is None:
            raise ValueError(
                f"Invalid machine '{machine}' for ASDLoader. The machine must have the form 'omi-<k>'."
            )
        self.machine = machine
        for file in self._files():
            if not os.path.isfile(file):
                raise FileNotFoundError(f"No such file: {file}")

    def _files(self) -> tuple[str, str, str]:
        return tuple(
            os.path.join(self.path, "processed", f"{self.machine}_{part}.pkl")
            for part in ("train", "test", "test_label")
        )

    def _load(self) -> DataSet:
        X_train, X_test, y_test = (_load_pickle(file) for file in self._files())
        return DataSet(
            X_test=np.asarray(X_test, dtype=float).reshape(len(X_test), -1),
            y_test=np.asarray(y_test, dtype=int),
            X_train=np.asarray(X_train, dtype=float).reshape(len(X_train), -1),
        )
