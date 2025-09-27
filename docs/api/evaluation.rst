Evaluation module
=================

.. automodule:: dtaianomaly.evaluation


Base Objects
------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Metric
    BinaryMetric
    ProbaMetric
    FBetaMixin

Affiliation-based
-----------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AffiliationPrecision
    AffiliationRecall
    AffiliationFBeta


Area Under the Curve (AUC)
--------------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AreaUnderPR
    AreaUnderROC


Classification-based
--------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Precision
    Recall
    FBeta


Compound metrics
----------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ThresholdMetric
    BestThresholdMetric


Event-wise
----------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EventWisePrecision
    EventWiseRecall
    EventWiseFBeta


Point-adjusted
--------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PointAdjustedPrecision
    PointAdjustedRecall
    PointAdjustedFBeta


Range-based
-----------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RangeBasedPrecision
    RangeBasedRecall
    RangeBasedFBeta


UCR Score
---------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    UCRScore


Volume Under the Surface (VUS)
------------------------------

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RangeAreaUnderPR
    RangeAreaUnderROC
    VolumeUnderPR
    VolumeUnderROC
