Installation
============

qecsim is available from the Python Package Index (PyPI) for installation using the Python package installer `pip`_.
Optionally, logging and a faster matching library can be configured.

.. _pip: https://pip.pypa.io/en/stable/quickstart/


Package
-------

qecsim requires Python 3.5+ and can be installed and upgraded using pip:

.. code-block:: bash

    $ pip install -U qecsim

A nice way to install qecsim is using a virtual environment:

.. code-block:: bash

    $ python3 --version             # qecsim requires Python 3.5+
    Python 3.7.7
    $ python3 -m venv venv          # create new virtual environment
    $ source venv/bin/activate      # activate venv (Windows: venv\Scripts\activate)
    (venv) $ pip install qecsim     # install qecsim
    ...
    Successfully installed ... qecsim-1.0b1 ...
    (venv) $
    (venv) $ qecsim --version       # verify qecsim cli
    qecsim, version 1.0b1
    (venv) $ deactivate             # deactivate venv
    $


Logging configuration (optional)
--------------------------------

By default, qecsim uses the logging configuration provided by :func:`logging.basicConfig`. Optionally, logging may be
configured using a file parsed with :func:`logging.config.fileConfig`. The following locations are searched in order:

* ``$QECSIM_CFG/logging_qecsim.ini``
* ``./logging_qecsim.ini``
* ``~/.qecsim/logging_qecsim.ini``

where ``QECSIM_CFG`` is an environment variable and ``~`` indicates the current user's home directory. An example
logging configuration file is available here: :download:`logging_qecsim.ini <../logging_qecsim.ini>`


.. _install_blossom:

Fast matching library (optional)
--------------------------------

TODO
