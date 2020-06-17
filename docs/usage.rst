Usage
=====

qecsim is available as a Python zip application with a simple command line interface that provides access to all
features. It can also be used as a library via the fully documented API.

Installation
------------

qecsim can be downloaded as a single zip file that requires no special installation.

It has the following dependencies:

*   Python 3.5+: https://www.python.org/
*   Numpy 1.17+: http://docs.scipy.org/doc/numpy/
*   Scipy 1.3+: https://docs.scipy.org/doc/ (only required for mps decoders)
*   Mpmath 1.0+: http://mpmath.org/ (only required for mps decoders)
*   NetworkX 2.1+: http://networkx.github.io/ (only required for mwpm decoders)
*   Click 7.0+: http://click.pocoo.org/ (only required for command-line use)

A nice way to install the dependencies is using a virtual environment:

.. code-block:: bash

    $ python3 --version  # check python version
    Python 3.7.7
    $ python3 -m venv myvenv  # create new virtual env
    $ source myvenv/bin/activate  # activate myvenv
    (myvenv) $ pip install numpy scipy mpmath networkx click  # install dependencies
    ...
    Successfully installed click-7.1.1 decorator-4.4.2 mpmath-1.1.0 networkx-2.4 numpy-1.18.2 scipy-1.4.1
    (myvenv) $
    (myvenv) $ python3 qecsim.pyz --version  # verify qecsim cli
    qecsim, version 0.17
    (myvenv) $ deactivate  # deactivate virtual env
    $

Command-line
------------

qecsim can be accessed via the command line:

.. runblock:: console

    $ qecsim --help

The ``merge`` command merges simulation data according to :func:`qecsim.app.merge`:

.. runblock:: console

    $ qecsim merge --help

The ``run`` command runs simulations according to :func:`qecsim.app.run`:

.. runblock:: console

    $ qecsim run --help

The ``run-ftp`` command runs fault-tolerant simulations according to :func:`qecsim.app.run_ftp`:

.. runblock:: console

    $ qecsim run-ftp --help

Library
-------

qecsim can be accessed as a library:

.. runblock:: pycon

    >>> import qecsim
    >>> qecsim.__version__

See doc:`demos` and doc:`api` for further information.