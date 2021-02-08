Usage
=====

qecsim provides access to all features via a command-line interface. It can also be used as a library via the
fully-documented API. It includes many common codes, error models and decoders and can be extended with additional
components.


Command-line
------------

qecsim can be accessed via the command line in two ways:

.. code-block:: bash

    $ qecsim                    # console script
    $ python3 -O -m qecsim      # module script with Python options e.g -O for optimize

The default command provides version and general help options:

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

qecsim can be extended with additional codes, error models and decoders that integrate into the command-line interface.
See https://github.com/qecsim/qecsimext for a basic example.
