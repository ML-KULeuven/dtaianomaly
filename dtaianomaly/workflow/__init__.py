"""
It is often crucial to perform large scale experiments to analyze the performance
of anomaly detectors on large benchmark sets. The ``Workflow`` module offer such
functionality and can be imported as follows:

>>> from dtaianomaly import workflow

We refer to the `documentation <https://dtaianomaly.readthedocs.io/en/stable/getting_started/examples/quantitative_evaluation.html>`_
for more information regarding the configuration and use of a Workflow.
"""

from ._Job import Job
from ._JobBasedWorkflow import JobBasedWorkflow
from ._Workflow import Workflow
from ._workflow_from_config import interpret_config, workflow_from_config

__all__ = [
    "Workflow",
    "Job",
    "JobBasedWorkflow",
    "workflow_from_config",
    "interpret_config",
]
