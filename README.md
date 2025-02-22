<p align="center">
<img width="300" src="docs/logo/readme.svg" alt=""/>
</p>

<h1 align="center">dtaianomaly</h1>
<h2 align="center">Time series anomaly detection</h2>

[![Unit tests](https://github.com/ML-KULeuven/dtaianomaly/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ML-KULeuven/dtaianomaly/actions/workflows/unit-tests.yml)
[![Doctest](https://github.com/ML-KULeuven/dtaianomaly/actions/workflows/doctest.yml/badge.svg)](https://github.com/ML-KULeuven/dtaianomaly/actions/workflows/doctest.yml)
[![docs-stable](https://img.shields.io/readthedocs/aeon-toolkit/latest?logo=readthedocs&label=docs%20%28stable%29)](https://dtaianomaly.readthedocs.io/en/stable)
[![docs-latest](https://img.shields.io/readthedocs/aeon-toolkit/latest?logo=readthedocs&label=docs%20%28latest%29)](https://dtaianomaly.readthedocs.io/en/latest)
[![PyPi Version](https://img.shields.io/pypi/v/dtaianomaly.svg)](https://pypi.org/project/dtaianomaly/)
[![Downloads](https://static.pepy.tech/badge/dtaianomaly)](https://pepy.tech/project/dtaianomaly)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/dtaianomaly)](https://pypi.python.org/pypi/dtaianomaly/)
[![PyPI license](https://img.shields.io/pypi/l/dtaianomaly.svg)](https://pypi.python.org/pypi/dtaianomaly/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


A simple-to-use Python package for the development and analysis of time series anomaly 
detection techniques. Here we describe the main usage of `dtaianomaly`, but be sure to
check out the [documentation](https://dtaianomaly.readthedocs.io/en/stable/index.html) 
for more information. 

## Installation

The preferred way to install `dtaianomaly` is via PyPi. See the [documentation](https://dtaianomaly.readthedocs.io/en/stable/index.html) 
for more options.
```
pip install dtaianomaly
```

## Features

The three key features of `dtaianomaly` are as follows:
1. State-of-the-art time series anomaly detection via a simple API.
   [Learn more!](https://dtaianomaly.readthedocs.io/en/stable/getting_started/examples/anomaly_detection.html)
2. Develop custom models for anomaly detection.
   [Learn more!](https://dtaianomaly.readthedocs.io/en/stable/getting_started/examples/custom_models.html)
3. Quantitative evaluation of time series anomaly detection.
   [Learn more!](https://dtaianomaly.readthedocs.io/en/stable/getting_started/examples/quantitative_evaluation.html)

## Example

Below code shows a simple example of `dtaianomaly`, which is part of the 
[anomaly detection notebook](notebooks/Anomaly-detection.ipynb). More examples 
are provided in the [other notebooks](notebooks) and in the 
[documentation](https://dtaianomaly.readthedocs.io/en/stable/getting_started/examples.html).

```python
from dtaianomaly.data import demonstration_time_series
from dtaianomaly.preprocessing import MovingAverage
from dtaianomaly.anomaly_detection import MatrixProfileDetector

# Load the data
X, y = demonstration_time_series()

# Preprocess the data using a moving average
preprocessor = MovingAverage(window_size=10)
X_, _ = preprocessor.fit_transform(X)

# Fit the matrix profile detector on the processed data
detector = MatrixProfileDetector(window_size=100)
detector.fit(X_)

# Compute either the decision scores, specific to the detector, or the anomaly probabilities
decision_scores = detector.decision_function(X_)
anomaly_probabilities = detector.predict_proba(X_)
```
![Demonstration-time-series-detected-anomalies.svg](https://github.com/ML-KULeuven/dtaianomaly/blob/main/notebooks/Demonstration-time-series-detected-anomalies.svg?raw=true)

## Acknowledgments

If you find ``dtaianomaly`` useful for your work, we would appreciate the following 
[citation](https://arxiv.org/abs/2502.14381):

```bibtex
@article{carpentier2025dtaianomalypythonlibrarytime,
      title={dtaianomaly: A Python library for time series anomaly detection}, 
      author={Louis Carpentier and Nick Seeuws and Wannes Meert and Mathias Verbeke},
      year={2025},
      eprint={2502.14381},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.14381}, 
}
```
> Carpentier, L., Seeuws, N., Meert, W., Verbeke, M.: dtaianomaly: A python 
> library for time series anomaly detection (2025), https://arxiv.org/abs/2502.14381

## Contribute

The goal of ``dtaianomaly`` is to be community-driven. All types of contributions
are welcome. This includes code, but also bug reports, improvements to the documentation,
additional tests and more. Check out [the documentation](https://dtaianomaly.readthedocs.io/en/stable/additional_information/contributing.html)
to find more information about how you can contribute!

## License

    Copyright (c) 2023-2025 KU Leuven, DTAI Research Group
    
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
