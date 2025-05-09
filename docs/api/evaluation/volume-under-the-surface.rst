Volume Under the Surface (VUS)
==============================

Implementation of the Volume Under the Surface (VUS) metrics proposed by :cite:`paparrizos2022volume`
The implementations are adopted from :cite:`wenig2022timeeval`, who slightly modified the original
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

In addition, we numbafied the most expensive part of the code (i.e., computing the recalls,
precisions and false positive rates for every threshold), which leads to a more than 25x
speedup on the demonstration time series.

.. autoclass:: dtaianomaly.evaluation.RangeAreaUnderPR
.. autoclass:: dtaianomaly.evaluation.RangeAreaUnderROC
.. autoclass:: dtaianomaly.evaluation.VolumeUnderPR
.. autoclass:: dtaianomaly.evaluation.VolumeUnderROC
