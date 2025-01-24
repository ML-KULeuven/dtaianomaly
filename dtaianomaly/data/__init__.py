"""
This module contains functionality to dynamically load data when 
executing a pipeline or workflow. It can be imported as follows:

>>> from dtaianomaly import data

Custom data loaders can be implemented by extending :py:class:`~dtaianomaly.data.LazyDataLoader`.
"""

from .LazyDataLoader import LazyDataLoader, from_directory
from .DataSet import DataSet
from .synthetic import make_sine_wave, demonstration_time_series
from .UCRLoader import UCRLoader

__all__ = [
    "LazyDataLoader",
    "DataSet",
    "from_directory",
    "demonstration_time_series",
    "make_sine_wave",
    "UCRLoader",
]
