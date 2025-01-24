"""
This module contains the workflow functionality.

>>> from dtaianomaly import workflow

Below we illustrate how a simple workflow can be initialized, which will
apply Matrix Profile and Isolation Forest on a dataset from the UCR
archive, and compute the area under the ROC and PR curves:

.. testsetup::

   import os
   os.chdir('..')


>>> from dtaianomaly.data import UCRLoader
>>> from dtaianomaly.anomaly_detection import MatrixProfileDetector, IsolationForest
>>> from dtaianomaly.evaluation import AreaUnderROC, AreaUnderPR
>>> workflow = workflow.Workflow(
...     dataloaders=[
...         UCRLoader(path='data/UCR-time-series-anomaly-archive/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt'),
...     ],
...     detectors=[MatrixProfileDetector(window_size=100), IsolationForest(15)],
...     metrics=[AreaUnderROC(), AreaUnderPR()]
... )

We refer to the `documentation <https://dtaianomaly.readthedocs.io/en/stable/getting_started/quantitative_evaluation.html>`_
for more information regarding the configuration and use of a Workflow.
"""

from .Workflow import Workflow
from .workflow_from_config import workflow_from_config, interpret_config

__all__ = ["Workflow", "workflow_from_config", "interpret_config"]
