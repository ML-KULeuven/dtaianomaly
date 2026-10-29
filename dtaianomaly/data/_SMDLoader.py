import os
import re
from pathlib import Path

import numpy as np

from dtaianomaly.data._DataSet import DataSet
from dtaianomaly.data._PathDataLoader import PathDataLoader

__all__ = ["SMDLoader"]

_MACHINE_PATTERN = re.compile(r"^machine-\d+-\d+$")


class SMDLoader(PathDataLoader):
    """
    Lazy dataloader for the Server Machine Dataset (SMD) :cite:`su2019robust`.

    The Server Machine Dataset is a 5-week-long dataset collected from a large
    internet company. It consists of the data of 28 machines, each with 38
    attributes. The data of each machine is split into a train and test set
    of equal size. Only the test set is labeled.

    Parameters
    ----------
    path : str
        The path to the root of the dataset, which contains the folders
        ``train``, ``test`` and ``test_label``.
    machine : str
        The name of the machine to load, of the form ``machine-<x>-<y>``.
    do_caching : bool, default=False
        Whether to cache the loaded data or not.

    Raises
    ------
    ValueError
        If ``machine`` does not have the form ``machine-<x>-<y>``.
    FileNotFoundError
        If the train, test or label file of the machine does not exist.

    Notes
    -----
    This implementation expects the files of the dataset to be organized as
    ``<path>/train/<machine>.txt``, ``<path>/test/<machine>.txt`` and
    ``<path>/test_label/<machine>.txt``. Because ``path`` points to the root of
    the dataset, :py:func:`~dtaianomaly.data.from_directory` can not be used
    with this loader. The data always has two dimensions.

    Examples
    --------
    >>> from dtaianomaly.data import SMDLoader
    >>> smd_data_set = SMDLoader("ServerMachineDataset", "machine-1-1").load()  # doctest: +SKIP
    """

    machine: str

    def __init__(self, path: str | Path, machine: str, do_caching: bool = False):
        super().__init__(path, do_caching)
        if not isinstance(machine, str) or _MACHINE_PATTERN.match(machine) is None:
            raise ValueError(
                f"Invalid machine '{machine}' for SMDLoader. The machine must have the form 'machine-<x>-<y>'."
            )
        self.machine = machine
        for file in self._files():
            if not os.path.isfile(file):
                raise FileNotFoundError(f"No such file: {file}")

    def _files(self) -> tuple[str, str, str]:
        return tuple(
            os.path.join(self.path, folder, f"{self.machine}.txt")
            for folder in ("train", "test", "test_label")
        )

    def _load(self) -> DataSet:
        train_file, test_file, test_label_file = self._files()
        return DataSet(
            X_test=np.loadtxt(test_file, delimiter=",", ndmin=2),
            y_test=np.loadtxt(test_label_file, dtype=int),
            X_train=np.loadtxt(train_file, delimiter=",", ndmin=2),
        )
