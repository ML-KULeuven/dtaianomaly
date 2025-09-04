import json


def load_configuration(path: str = None) -> dict:
    if path is None:
        return load_default_configuration()
    else:
        with open(path, "r") as f:
            return json.load(f)


def load_default_configuration() -> dict:
    return {
        "data-loader": {"default": "DemonstrationTimeSeriesLoader", "exclude": []},
        "detector": {
            "default": "IsolationForest",
            "exclude": [
                "MultivariateDetector",
                "AlwaysNormal",
                "AlwaysAnomalous",
                "RandomDetector",
            ],
            "parameters-required": {"window_size": 64, "neighborhood_size_before": 64},
            "parameters-optional": {
                "window_size_selection": {
                    "label": "Window size",
                    "options": [
                        ["Manual", "Manual"],
                        ["Dominant Fourier Frequency", "fft"],
                        ["Highest Autocorrelation", "acf"],
                        ["Summary Statistics Subsequence", "suss"],
                        ["Multi-Window-Finder", "mwf"],
                    ],
                    "help": "The used method for setting the window size. Options are:\n\n"
                    "- **Manual:** Manually set the window size to a specific size.\n"
                    "- **Dominant Fourier Frequency:** Use the window size which corresponds to the dominant frequency in the Fourier domain.\n"
                    "- **Highest Autocorrelation:** Use the window size which corresponds to maximum autocorrelation.\n"
                    "- **Summary Statistics Subsequence:** Find a window size such that the statics of that window are similar to those of the full time series. \n"
                    "- **Multi-Window-Finder:** Find a window size such that the moving average is small. ",
                },
                "window_size": {
                    "label": "Manual window size",
                    "type": "number_input",
                    "min_value": 1,
                    "step": 1,
                    "value": 64,
                    "help": "The manually-set size of the sliding window.",
                },
                "stride": {
                    "label": "Stride",
                    "type": "number_input",
                    "min_value": 1,
                    "step": 1,
                    "value": 1,
                    "help": "The stride of a sliding window is the number of steps the window moves forward each time.",
                },
                "start_level": {
                    "type": "number_input",
                    "label": "Start level",
                    "min_value": 0,
                    "value": 3,
                    "step": 1,
                    "help": "The first level for computing the Discrete Wavelet Transform.",
                },
                "quantile_epsilon": {
                    "type": "slider",
                    "label": "Quantile",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                    "value": 0.01,
                    "help": "The percentile used as threshold on the likelihood estimates.",
                },
                "padding_mode": {
                    "type": "selectbox",
                    "label": "Padding",
                    "options": ["wrap", "symmetric"],
                    "index": 0,
                    "help": "How the time series is padded:"
                    "\n-**wrap:** Use the first values to pad at the end and the last values to pad at the beginning."
                    "\n- **symmetric:** Pads with the reflection of the time series.",
                },
                "sequence_length_multiplier": {
                    "type": "number_input",
                    "label": "Sequence length multiplier",
                    "min_value": 1.0,
                    "value": 4.0,
                    "step": 0.1,
                    "help": "The amount by which the window size should be multiplied to create sliding windows for clustering the data using KShape.",
                },
                "overlap_rate": {
                    "type": "slider",
                    "label": "Overlap rate",
                    "min_value": 0.01,
                    "max_value": 1.0,
                    "step": 0.01,
                    "value": 0.5,
                    "help": "The overlap of the sliding windows for clustering the data. Will be used to compute a relative stride to avoid trivial matches when clustering subsequences.",
                },
                "normalize": {
                    "type": "toggle",
                    "label": "Z-scale",
                    "value": True,
                    "help": "Whether to Z-scale the time series.",
                },
                "p": {
                    "type": "number_input",
                    "label": "Norm",
                    "min_value": 1.0,
                    "value": 2.0,
                    "step": 0.1,
                    "help": "The used norm for computing the distances with the matrix profile.",
                },
                "k": {
                    "type": "number_input",
                    "label": "K",
                    "min_value": 1,
                    "value": 1,
                    "step": 1,
                    "help": "Use the distance to the K-th nearest neighbor as an anomaly score.",
                },
                "novelty": {
                    "type": "toggle",
                    "label": "Novelty detection",
                    "value": False,
                    "help": "If novelty detection should be performed, i.e., detect anomalies with regards to a normal time series.",
                },
                "neighborhood_size_before": {
                    "type": "number_input",
                    "label": "Neighborhood size before the sample",
                    "min_value": 1,
                    "value": 64,
                    "step": 1,
                    "help": "The number of observations before the sample to include in the neighborhood.",
                },
                "neighborhood_size_after": {
                    "type": "number_input",
                    "label": "Neighborhood size after the sample",
                    "min_value": 0,
                    "value": None,
                    "step": 1,
                    "help": "The number of observations after the sample to include in the neighborhood. If no value "
                    "is given, the neighborhood size after the window will be set to the same value as the "
                    "neighborhood size before the window.",
                },
                "seed": {
                    "type": "number_input",
                    "label": "Seed",
                    "min_value": 1,
                    "value": None,
                    "step": 1,
                    "help": "The random seed to set.",
                },
                "max_iter": {
                    "type": "number_input",
                    "label": "Maximum number of iterations",
                    "min_value": 1,
                    "value": 1000,
                    "step": 1,
                    "help": "The maximum number of iterations to perform during optimization.",
                },
                "error_metric": {
                    "type": "selectbox",
                    "label": "Anomaly score metric",
                    "options": ["mean-absolute-error", "mean-squared-error"],
                    "index": 0,
                    "help": "The error measure to use as anomaly scores between the predicted values and the actually observed values in the time series.",
                },
                "latent_space_dimension": {
                    "type": "number_input",
                    "label": "Dimension of the latent space",
                    "min_value": 1,
                    "value": 32,
                    "step": 1,
                    "help": "The dimension of the latent space of the auto encoder, i.e., the number of neurons.",
                },
                "dropout_rate": {
                    "type": "slider",
                    "label": "Dropout rate",
                    "help": "The drop out rate to use within the network, i.e., the percentage of weights that are frozen during training.",
                    "min_value": 0.0,
                    "max_value": 0.99,
                    "step": 0.01,
                    "value": 0.0,
                },
                "activation_function": {
                    "type": "selectbox",
                    "label": "Activation function",
                    "options": ["linear", "relu", "sigmoid", "tanh"],
                    "index": 1,
                    "help": "The activation function to use for including non-linearity in the network.",
                },
                "batch_normalization": {
                    "type": "toggle",
                    "label": "Apply batch normalization",
                    "value": True,
                    "help": "Whether to add batch normalization after each layer or not.",
                },
                "standard_scaling": {
                    "type": "toggle",
                    "label": "Apply standard scaling",
                    "value": True,
                    "help": "Whether to apply standard scaling to each window before feeding it to the neural network.",
                },
                "batch_size": {
                    "type": "number_input",
                    "label": "Batch size",
                    "min_value": 1,
                    "value": 32,
                    "step": 1,
                    "help": "The batch size to use for training the network, i.e., the number of samples to feed simultaneously for computing the loss and updating the weights.",
                },
                "loss": {
                    "type": "selectbox",
                    "label": "Loss function",
                    "options": ["mse", "l1", "huber"],
                    "index": 0,
                    "help": "The loss function to use when training the network. Options are:"
                    "\n-**mse:** Use the Mean Squared Error loss."
                    "\n-**l1:** Use the L1-loss or the mean absolute error."
                    "\n-**huber:** Use the huber loss, which smoothly combines the MSE-loss with the L1-loss.",
                },
                "optimizer": {
                    "type": "selectbox",
                    "label": "Optimizer",
                    "options": ["adam", "sgd"],
                    "index": 0,
                    "help": "The optimizer to use for updating the weights during training.",
                },
                "learning_rate": {
                    "type": "select_slider",
                    "label": "Learning rate",
                    "options": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                    "value": 1e-3,
                    "help": "The number of training iterations.",
                },
                "n_epochs": {
                    "type": "number_input",
                    "label": "Number of epochs",
                    "min_value": 1,
                    "value": 10,
                    "step": 1,
                    "help": "The number of training iterations.",
                },
                "forecast_length": {
                    "type": "number_input",
                    "label": "Forecast length",
                    "min_value": 1,
                    "value": 1,
                    "step": 1,
                    "help": "The number of observations in time that should be forecasted for each window.",
                },
                "n_clusters": {
                    "type": "number_input",
                    "label": "Number of clusters",
                    "min_value": 1,
                    "value": 4,
                    "step": 1,
                    "help": "The number of clusters to use in the clustering algorithm.",
                },
                "alpha": {
                    "HistogramBasedOutlierScore": {
                        "type": "slider",
                        "label": "Alpha",
                        "help": "Parameter used for regularization and for preventing overflow.",
                        "min_value": 0.01,
                        "max_value": 0.99,
                        "step": 0.01,
                        "value": 0.1,
                    },
                    "ClusterBasedLocalOutlierFactor": {
                        "type": "slider",
                        "label": "Alpha",
                        "help": "The ratio for deciding small and large clusters.",
                        "min_value": 0.5,
                        "max_value": 1.0,
                        "step": 0.01,
                        "value": 0.9,
                    },
                },
                "beta": {
                    "type": "number_input",
                    "label": "Beta",
                    "min_value": 1.0,
                    "value": 5.0,
                    "step": 0.1,
                    "help": "Parameter used for splitting the clusters in 'large' and 'small' clusters.",
                },
                "kernel_size": {
                    "type": "number_input",
                    "label": "Kernel size",
                    "min_value": 1,
                    "value": 3,
                    "step": 1,
                    "help": "The size each kernel in the convolutional layers should have.",
                },
                "n_bins": {
                    "type": "number_input",
                    "label": "Number of bins",
                    "min_value": 1,
                    "value": 10,
                    "step": 1,
                    "help": "The number of bins to use for each feature.",
                },
                "tol": {
                    "type": "slider",
                    "label": "Tolerance",
                    "help": "Parameter defining the flexibility for dealing with samples that fall outside the bins.",
                    "min_value": 0.01,
                    "max_value": 0.99,
                    "step": 0.01,
                    "value": 0.5,
                },
                "n_estimators": {
                    "type": "number_input",
                    "label": "Number of estimators",
                    "min_value": 1,
                    "value": 100,
                    "step": 1,
                    "help": "The number of base trees to include in the ensemble.",
                },
                "max_samples": {
                    "type": "number_input",
                    "label": "Maximum number of samples",
                    "help": "The number of samples to use for learning each base estimator. ",
                    "min_value": 1,
                    "step": 1,
                    "value": 256,
                },
                "max_features": {
                    "type": "slider",
                    "label": "Maximum number of features",
                    "help": "The maximum number of features to use for training each base estimator. This value represents the percentage of features to use.",
                    "min_value": 0.01,
                    "max_value": 1.0,
                    "step": 0.01,
                    "value": 1.0,
                },
                "n_neighbors": {
                    "type": "number_input",
                    "label": "Number of neighbors",
                    "help": "The number of neighbors to include when computing anomaly scores",
                    "min_value": 1,
                    "step": 1,
                    "value": 10,
                },
                "method": {
                    "type": "selectbox",
                    "label": "Handling nearest neighbors",
                    "options": ["largest", "mean", "median"],
                    "index": 0,
                    "help": "Method for computing the anomaly scores based on the nearest neighbors. Valid options are:"
                    "\n-**largest:** Use the distance to the kth neighbor."
                    "\n-**mean:** Use the mean distance to the nearest neighbors"
                    "\n-**median:** Use the median distance to the nearest neighbors",
                },
                "metric": {
                    "type": "selectbox",
                    "label": "Distance metric",
                    "options": [
                        "minkowski",
                        "euclidean",
                        "jaccard",
                        "hamming",
                        "mahalanobis",
                        "cosine",
                        "correlation",
                    ],
                    "index": 0,
                    "help": "The distance metric to use when computing the distances between samples.",
                },
                "n_components": {
                    "type": "slider",
                    "label": "Percentage of components",
                    "help": "The percentage of PCA-components to use",
                    "min_value": 0.01,
                    "max_value": 1.0,
                    "step": 0.01,
                    "value": 1.0,
                },
                "kernel": {
                    "type": "selectbox",
                    "label": "Kernel",
                    "options": ["linear", "poly", "rbf", "sigmoid", "cosine"],
                    "index": 2,
                    "help": "The kernel to use to map the data into a new space.",
                },
                "hidden_units": {
                    "type": "number_input",
                    "label": "Number of hidden units",
                    "help": "The number of hidden units (i.e., LSTM-cells) to include in each LSTM-layer.",
                    "min_value": 1,
                    "step": 1,
                    "value": 8,
                },
                "num_lstm_layers": {
                    "type": "number_input",
                    "label": "Number of LSTM-layers",
                    "help": "The number of LSTM-layers to include in the network.",
                    "min_value": 1,
                    "step": 1,
                    "value": 1,
                },
                "bias": {
                    "type": "toggle",
                    "label": "Include bias",
                    "value": True,
                    "help": "Whether to include a learnable bias at the end of each layer.",
                },
                "num_heads": {
                    "type": "number_input",
                    "label": "Number of heads",
                    "help": "The number of attention heads to include in each attention-layer.",
                    "min_value": 1,
                    "step": 1,
                    "value": 12,
                },
                "num_transformer_layers": {
                    "type": "number_input",
                    "label": "Number of attention layers",
                    "help": "The number of attention layers to include in the transformer.",
                    "min_value": 1,
                    "step": 1,
                    "value": 1,
                },
                "dimension_feedforward": {
                    "type": "number_input",
                    "label": "Dimension of the feed forward layer",
                    "help": "The dimension of the linear layer at the end of each attention layer.",
                    "min_value": 1,
                    "step": 1,
                    "value": 32,
                },
            },
        },
        "metric": {
            "default": ["EventWiseFBeta", "VolumeUnderPrCurve"],
            "exclude": ["BestThresholdMetric", "ThresholdMetric"],
            "parameters-required": {"cutoff": 0.9},
            "parameters-optional": {
                "cutoff": {
                    "type": "slider",
                    "label": "Cutoff",
                    "help": "The cutoff for converting the anomaly scores to a binary prediction. The cutoff is done on the predicted anomaly scores after min-max scaling.",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                    "value": 0.9,
                },
                "beta": {
                    "type": "number_input",
                    "label": "Beta",
                    "min_value": 0.0,
                    "step": 0.01,
                    "value": 1.0,
                    "help": "Determines the weight of recall in the combined score.",
                },
                "buffer_size": {
                    "type": "number_input",
                    "label": "Buffer size",
                    "min_value": 1,
                    "step": 1,
                    "value": 100,
                    "help": "Size of the buffer region around an anomaly. Half of the buffer added before the anomalous event and half of the buffer is added after the anomaly.",
                },
                "compatibility_mode": {
                    "type": "toggle",
                    "label": "Use original version",
                    "value": False,
                    "help": "Whether to use the originally proposed version of this metric or the implementation of TimeEval:"
                    "\n- For the recall (FPR) existence reward, anomalies are counted as separate events, even if the added slopes overlap;"
                    "\n- Overlapping slopes don't sum up in their anomaly weight, the anomaly weight for each point in the ground truth is maximized;"
                    "\n- The original slopes are asymmetric: the slopes at the end of anomalies are a single point shorter than the ones at the beginning of anomalies. Symmetric slopes are used, with the same size for the beginning and end of anomalies;"
                    "\n- A linear approximation of the slopes is used instead of the convex slope shape presented in the paper.",
                },
                "max_samples": {
                    "type": "number_input",
                    "label": "Maximum number of thresholds",
                    "value": 100,
                    "min_value": 1,
                    "step": 1,
                    "help": "The number of thresholds to put on the anomaly scores. This offers a trade-off between exactness of the metric and computation time.",
                },
                "alpha": {
                    "type": "slider",
                    "label": "Alpha",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "value": 0.5,
                    "step": 0.01,
                    "help": "The importance of detecting the events (even if it is only a single detected point) compared to detecting a large portion of the ground truth events.",
                },
                "delta": {
                    "type": "selectbox",
                    "label": "Delta",
                    "options": ["flat", "front", "back", "middle"],
                    "index": 0,
                    "help": "Bias for the position of the predicted anomaly in the ground truth anomalous range:"
                    "\n- **flat:** Equal bias towards all positions in the ground truth anomalous range."
                    "\n- **front:** Predictions that are near the front of the ground truth anomaly (i.e. early detection) have a higher weight."
                    "\n- **back:** Predictions that are near the end of the ground truth anomaly (i.e. late detection) have a higher weight."
                    "\n- **middle:** Predictions that are near the center of the ground truth anomaly have a higher weight.",
                },
                "gamma": {
                    "type": "selectbox",
                    "label": "Gamma",
                    "options": ["one", "reciprocal"],
                    "index": 0,
                    "help": "Penalization approach for detecting multiple ranges with a single range: "
                    "\n- **one:** Fragmented detection should not be penalized."
                    "\n- **reciprocal:** Weight fragmented detection of $N$ ranges with as single range by a factor of $1/N$.",
                },
                "max_buffer_size": {
                    "type": "number_input",
                    "label": "Maximum buffer size",
                    "min_value": 1,
                    "step": 1,
                    "value": 250,
                    "help": "Maximum size of the buffer region around an anomaly. Half of the buffer added before the anomalous event and half of the buffer is added after the anomaly. The metric iterates over all the buffer sizes to to create a volume.",
                },
            },
        },
    }
