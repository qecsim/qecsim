Usage
=====

qecsim includes a simple command line interface that provides access to all features. It can also be used as a library
via the fully documented API.

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

Extension
---------

TODO
