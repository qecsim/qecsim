Installation
============

qecsim is available from the Python Package Index for installation using the Python package installer pip. Optionally,
logging and a faster matching library can be configured.

Package
-------

The qecsim package requires Python 3.5+ and can be installed directly from PyPI:

.. code-block:: bash

    $ pip install qecsim

A nice way to install qecsim is using a virtual environment:

.. code-block:: bash

    $ python3 --version  # check python version
    Python 3.7.7
    $ python3 -m venv myvenv  # create new virtual env
    $ source myvenv/bin/activate  # activate myvenv
    (myvenv) $ pip install qecsim  # install qecsim
    ...
    Successfully installed click-7.1.2 decorator-4.4.2 mpmath-1.1.0 networkx-2.4 numpy-1.19.0 qecsim-1.0a1 scipy-1.5.0
    (myvenv) $
    (myvenv) $ qecsim --version  # verify qecsim cli
    qecsim, version 1.0a1
    (myvenv) $ deactivate  # deactivate virtual env
    $


Logging configuration (optional)
--------------------------------

TODO

Fast matching library (optional)
--------------------------------

TODO
