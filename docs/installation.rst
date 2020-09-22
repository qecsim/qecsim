Installation
============

qecsim is available (TODO: COMING SOON) from the Python Package Index (PyPI) for installation using the Python package
installer `pip`_. Optionally, logging and a faster matching library can be configured.

.. _pip: https://pip.pypa.io/en/stable/quickstart/


Package
-------

qecsim requires Python 3.5+ and can be installed and upgraded using pip:

.. code-block:: bash

    $ pip install -U qecsim         # TODO: COMING SOON

A nice way to install qecsim is using a virtual environment:

.. code-block:: bash

    $ python3 --version             # qecsim requires Python 3.5+
    Python 3.7.8
    $ python3 -m venv venv          # create new virtual environment
    $ source venv/bin/activate      # activate venv (Windows: venv\Scripts\activate)
    (venv) $ pip install qecsim     # install qecsim
    ...
    Successfully installed ... qecsim-1.0b3 ...
    (venv) $
    (venv) $ qecsim --version       # verify qecsim cli
    qecsim, version 1.0b3
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

Several decoders included with qecsim use minimum-weight perfect matching in graphs via :func:`qecsim.graphtools.mwpm`.
By default, qecsim will use a Python matching implementation provided by NetworkX_. Optionally, qecsim can be configured
to use `Blossom V`_, a fast C++ matching implementation, due to Vladimir Kolmogorov:

    Vladimir Kolmogorov. "Blossom V: A new implementation of a minimum cost perfect matching algorithm."
    In *Mathematical Programming Computation* (MPC), July 2009, 1(1):43-67.

.. _NetworkX: https://networkx.github.io/
.. _Blossom V: http://pub.ist.ac.at/~vnk/software.html

The licence for Blossom V does not permit public redistribution of the code or its derivatives (see `Blossom V`_ and the
Blossom V README.TXT file for full details of the license and citation requirements). Therefore, Blossom V is not
packaged with qecsim.

If your use case satisfies the license requirements of Blossom V, you can configure qecsim to use Blossom V as follows:

* Download Blossom V ``blossom5-v2.05.src.tar.gz`` from http://pub.ist.ac.at/~vnk/software.html
* Download the Python wrapper :download:`blossom5-v2.05.pyw.tar.gz <../clib/blossom5-v2.05.pyw.tar.gz>`
* Build C++ library ``libpypm.so`` from Blossom V and the Python wrapper.

.. code-block:: bash

    $ tar -xzf blossom5-v2.05.src.tar.gz
    $ tar -xzf blossom5-v2.05.pyw.tar.gz
    $ cp blossom5-v2.05.pyw/* blossom5-v2.05.src/
    $ cd blossom5-v2.05.src/
    $ make -f Makefile-pyw
    ...
    c++ -shared ... -o libpypm.so


* Copy ``libpypm.so`` to one of the following locations, which are searched in order by qecsim:

    * ``$QECSIM_CFG/clib/libpypm.so``
    * ``./clib/libpypm.so``
    * ``~/.qecsim/clib/libpypm.so``

  where ``QECSIM_CFG`` is an environment variable and ``~`` indicates the current user's home directory.

* Check that Blossom V is available to qecsim:

.. code-block:: pycon

    >>> from qecsim.graphtools import blossom5
    >>> blossom5.available()
    True

The above procedure has been tested on Linux and MacOS. The Blossom V README.TXT file states that it should compile with
the Microsoft Visual C++ compiler, therefore the above procedure, with some adaptation, *should* work on Windows.
