Visualization module
====================

.. automodule:: dtaianomaly.visualization

.. plot::
   :context: reset
   :include-source: False

   import matplotlib.pyplot as plt
   plt.rcParams.update({
       'figure.autolayout': True,
       'figure.titlesize': 18
   })


.. autofunction:: dtaianomaly.visualization.plot_demarcated_anomalies

.. plot::
   :context: close-figs

   >>> from dtaianomaly.data import demonstration_time_series
   >>> from dtaianomaly.visualization import plot_demarcated_anomalies
   >>> X, y = demonstration_time_series()
   >>> fig = plot_demarcated_anomalies(X, y, figsize=(10, 3))
   >>> fig.suptitle("Example of 'plot_demarcated_anomalies'")  # doctest: +SKIP


.. autofunction:: dtaianomaly.visualization.plot_time_series_colored_by_score

.. plot::
   :context: close-figs

   >>> from dtaianomaly.data import demonstration_time_series
   >>> from dtaianomaly.visualization import plot_time_series_colored_by_score
   >>> X, y = demonstration_time_series()
   >>> fig = plot_time_series_colored_by_score(X, y, figsize=(10, 3))
   >>> fig.suptitle("Example of 'plot_time_series_colored_by_score' on the ground truth")  # doctest: +SKIP

.. plot::
   :context: close-figs

   >>> from dtaianomaly.data import demonstration_time_series
   >>> from dtaianomaly.visualization import plot_time_series_colored_by_score
   >>> from dtaianomaly.anomaly_detection import IsolationForest
   >>> X, _ = demonstration_time_series()
   >>> y_pred = IsolationForest(window_size=100).fit(X).predict_proba(X)
   >>> fig = plot_time_series_colored_by_score(X, y_pred, figsize=(10, 3))
   >>> fig.suptitle("Example of 'plot_time_series_colored_by_score' on predictions")  # doctest: +SKIP


.. autofunction:: dtaianomaly.visualization.plot_anomaly_scores

.. plot::
   :context: close-figs

   >>> from dtaianomaly.data import demonstration_time_series
   >>> from dtaianomaly.visualization import plot_anomaly_scores, plot_time_series_colored_by_score
   >>> from dtaianomaly.anomaly_detection import IsolationForest
   >>> X, y = demonstration_time_series()
   >>> y_pred = IsolationForest(window_size=100).fit(X).predict_proba(X)
   >>> fig = plot_anomaly_scores(X, y, y_pred, figsize=(10, 3), method_to_plot=plot_time_series_colored_by_score)
   >>> fig.suptitle("Example of 'plot_anomaly_scores'")  # doctest: +SKIP

.. plot::
   :context: close-figs

   >>> from dtaianomaly.data import demonstration_time_series
   >>> from dtaianomaly.visualization import plot_anomaly_scores, plot_time_series_colored_by_score
   >>> from dtaianomaly.anomaly_detection import IsolationForest
   >>> X, y = demonstration_time_series()
   >>> detector = IsolationForest(window_size=100).fit(X)
   >>> y_pred = detector.predict_proba(X)
   >>> confidence = detector.predict_confidence(X)
   >>> fig = plot_anomaly_scores(X, y, y_pred, confidence=confidence, figsize=(10, 3), method_to_plot=plot_time_series_colored_by_score)
   >>> fig.suptitle("Example of 'plot_anomaly_scores' with confidence ranges")  # doctest: +SKIP


.. autofunction:: dtaianomaly.visualization.plot_time_series_anomalies

.. plot::
   :context: close-figs

   >>> from dtaianomaly.data import demonstration_time_series
   >>> from dtaianomaly.visualization import plot_time_series_anomalies
   >>> from dtaianomaly.anomaly_detection import IsolationForest
   >>> from dtaianomaly.thresholding import FixedCutoff
   >>> X, _ = demonstration_time_series()
   >>> y_pred = IsolationForest(window_size=100).fit(X).predict_proba(X)
   >>> y_pred_binary = FixedCutoff(cutoff=0.9).threshold(y_pred)
   >>> fig = plot_time_series_anomalies(X, y, y_pred_binary, figsize=(10, 3))
   >>> fig.suptitle("Example of 'plot_time_series_anomalies'")  # doctest: +SKIP


.. autofunction:: dtaianomaly.visualization.plot_with_zoom

.. plot::
   :context: close-figs

   >>> from dtaianomaly.data import demonstration_time_series
   >>> from dtaianomaly.visualization import plot_with_zoom
   >>> X, y = demonstration_time_series()
   >>> fig = plot_with_zoom(X, y, start_zoom=700, end_zoom=1200, figsize=(10, 3))
   >>> fig.suptitle("Example of 'plot_with_zoom'")  # doctest: +SKIP
