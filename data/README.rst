Time series anomaly detection benchmarks
========================================

In this page, we describe all datasets that can be loaded with 
``dtaianomaly``. The table below provides a brief summary of these 
datasets. In addition, ``dtaianomaly`` provides the ability to load
custom time series data and synthetic data. How to do this is also
described on this page.

.. list-table::
   :header-rows: 1

   * - Dataset
     - Size
     - Download

   * - UCR
     - 500MB
     - `UCR Time Series Anomaly Archive <UCR_>`_

.. _UCR: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip

.. note::
    You can also read custom data by implementing a custom :py:class:`~dtaianomaly.data.LazyDataLoader`,
    as described in the `documentation <https://dtaianomaly.readthedocs.io/en/stable/index.html>`_.

Synthetic data
--------------

Within ``dtaianomaly``, it is possible to generate synthetic data for testing purposes.
First of all, it is possible to load the demonstration time series used throughout the
documentation of ``dtaianomaly``. This is done as follows:

>>> from dtaianomaly.data import demonstration_time_series
>>> X, y = demonstration_time_series()

Alternatively, it is possible to generate a synthetic sine wave with specified amplitude,
frequency, noise, ... via the :py:func:`dtaianomaly.data.make_sine_wave` method.

UCR time series anomaly archive
-------------------------------

The UCR time series anomaly archive consists of 250 time series, which have been published
to `mitigate` common issues in existing time series anomaly detection benchmarks :cite:`wu2023current`:

1. **Triviality**: many benchmarks are easily solved without any fancy algorithms;
2. **Unrealistic anomaly density**: the number of ground truth anomalies is relatively high, even though anomalies should be rare observations;
3. **Mislabeling**: the ground truth labels might not be perfectly aligned with the actual anomalies in the data;
4. **Run-to-failure bias**: most anomalies are located near the end of the time series.
