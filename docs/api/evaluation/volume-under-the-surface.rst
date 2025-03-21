Volume Under the Surface (VUS)
==============================

Implementation of the Volume Under the Surface (VUS) metrics proposed by [paparrizos2022volume]_
The implementations are adopted from [wenig2022timeeval]_, who slightly modified the original
implementations:

- For the recall (FPR) existence reward, anomalies are counted as separate events, even
  if the added slopes overlap;
- Overlapping slopes don't sum up in their anomaly weight, the anomaly weight for each
  point in the ground truth is maximized;
- The original slopes are asymmetric: the slopes at the end of anomalies are a single
  point shorter than the ones at the beginning of anomalies. Symmetric slopes are used,
  with the same size for the beginning and end of anomalies;
- A linear approximation of the slopes is used instead of the convex slope shape presented
  in the paper.

By default, the adjusted versions of each metric are used. To use the original implementations,
you can set ``compatibility_mode=True`` when initializing the metrics.

.. [paparrizos2022volume] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay,
   Aaron Elmore, and Michael J. Franklin. Volume under the surface: a new accuracy evaluation
   measure for time-series anomaly detection. Proceedings of the VLDB Endowment 15.11 (2022):
   2774-2787, doi: `10.14778/3551793.3551830 <https://doi.org/10.14778/3551793.3551830>`_.

.. [wenig2022timeeval] Phillip Wenig, Sebastian Schmidl, and Thorsten Papenbrock. TimeEval:
   A Benchmarking Toolkit for Time Series Anomaly Detection Algorithms. PVLDB, 15(12):
   3678 - 3681, 2022. doi:`10.14778/3554821.3554873 <https://doi.org/10.14778/3554821.3554873>`_

.. autoclass:: dtaianomaly.evaluation.RangeAreaUnderPR
.. autoclass:: dtaianomaly.evaluation.RangeAreaUnderROC
.. autoclass:: dtaianomaly.evaluation.VolumeUnderPR
.. autoclass:: dtaianomaly.evaluation.VolumeUnderROC
