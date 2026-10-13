Installation
============

From PyPi
---------

The preferred way to install ``dtaianomaly`` is through `PyPI <https://pypi.org/project/dtaianomaly/>`_.
Use the following command to install the latest version:

.. code-block:: bash

    pip install dtaianomaly

Some dependencies are optional to ensure that ``dtaianomaly`` remains
lightweight. These can be installed as extras, by adding the name of the
extra between square brackets:

.. code-block:: bash

    pip install "dtaianomaly[tqdm]"

The following extras are available:

- ``tqdm``: Install TQDM (for showing ``WorkFlow``-progress).
- ``chronos``: Install autogluon.timeseries, necessary for running Chronos.
- ``moment``: Install momentfm, necessary for running MOMENT (only Python 3.11).
- ``time_moe``: Install transformers 4.40.x and accelerate, necessary for running Time-MoE.

.. warning::

    The extra ``moment`` cannot be installed together with ``chronos`` or ``time_moe``,
    because momentfm requires an older, incompatible version of ``transformers``.
    Use a separate virtual environment for MOMENT. The extras ``chronos`` and
    ``time_moe`` can be combined, but this installs an older version of
    autogluon.timeseries (because Time-MoE requires ``transformers`` 4.40.x).

If you use `uv <https://docs.astral.sh/uv/>`_, the extras can be added to
your project in the same way:

.. code-block:: bash

    uv add "dtaianomaly[chronos]"

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