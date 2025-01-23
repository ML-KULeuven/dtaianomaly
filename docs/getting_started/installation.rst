Installation
============

From PyPi
---------

The preferred way to install ``dtaianomaly`` is through `PyPI <https://pypi.org/project/dtaianomaly/>`_.
Use the following command to install the latest version:

.. code-block:: bash

    pip install dtaianomaly

Some dependencies are optional to ensure that ``dtaianomaly`` remains
lightweight. Use the following command to install all optional
dependencies:

.. code-block:: bash

    pip install dtaianomaly[all]

It is also possible to only install a subset of the optional dependencies.
You can install these by replacing 'all' with the corresponding name in the
command above. To install multiple subsets, separate the names with a comma.
Currently, following subsets are available:

- ``tests``: Dependencies for running the tests.
- ``docs``: Dependencies for generating the documentation.
- ``notebooks``: Dependencies for using jupyter notebooks.

To install version ``X.Y.Z``, use the following command:

.. code-block:: bash

    pip install dtaianomaly==X.Y.Z


From GitHub
-----------

It is also possible to install ``dtaianomaly`` directly from `GitHub`_:

.. code-block:: bash

    pip install git+https://github.com/ML-KULeuven/dtaianomaly

Note that this will install the latest, *unreleased* version. Similarly as installation
through PyPi, it is also possible to install version ``X.Y.Z`` as follows:

.. code-block:: bash

    pip install git+https://github.com/ML-KULeuven/dtaianomaly@X.Y.Z

See the `release page <https://github.com/ML-KULeuven/dtaianomaly/releases>`_
for more information regarding the different versions.


From source
-----------

Finally, it is also possible to install ``dtaianomaly`` from the source code. First
download the source from `GitHub`_.
It is also possible to download the source code for a specific release on
`this webpage <https://github.com/ML-KULeuven/dtaianomaly/releases>`_.
Unzip the files, and navigate to the root directory of the repository in the terminal.
Then, ``dtaianomaly`` can be installed through the following command:

.. code-block:: bash

    pip install .

.. _GitHub: https://github.com/ML-KULeuven/dtaianomaly