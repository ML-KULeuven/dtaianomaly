Preprocessing module
====================

.. automodule:: dtaianomaly.preprocessing

Base objects
------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Preprocessor


Scaling
-------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MinMaxScaler
    StandardScaler
    RobustScaler


Smoothing
---------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    MovingAverage
    ExponentialMovingAverage
    Differencing
    PiecewiseAggregateApproximation


Under-sampling
--------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SamplingRateUnderSampler
    NbSamplesUnderSampler


Other preprocessors
-------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ChainedPreprocessor
    Identity
