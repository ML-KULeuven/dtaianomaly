"""
This module contains functionality to dynamically load data when
executing a pipeline or workflow. It can be imported as follows:

>>> from dtaianomaly import data

Custom data loaders can be implemented by extending :py:class:`~dtaianomaly.data.LazyDataLoader`.
"""

from ._CustomDataLoader import CustomDataLoader
from ._DataSet import DataSet
from ._DemonstrationTimeSeriesLoader import (
    DemonstrationTimeSeriesLoader,
    demonstration_time_series,
)
from ._LazyDataLoader import LazyDataLoader
from ._PathDataLoader import PathDataLoader, from_directory
from ._UCRLoader import UCRLoader

__all__ = [
    "LazyDataLoader",
    "PathDataLoader",
    "DataSet",
    "from_directory",
    "demonstration_time_series",
    "DemonstrationTimeSeriesLoader",
    "UCRLoader",
    "CustomDataLoader",
]
