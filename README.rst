qecsim
======

**qecsim** is a Python 3 package for simulating quantum error correction using stabilizer codes.

It includes many common codes, error models and decoders and can be extended with additional components.


Installation
------------

Install and update using `pip`_:

.. code-block:: text

    $ pip install -U qecsim

.. _pip: https://pip.pypa.io/en/stable/quickstart/


Usage
-----

qecsim includes a simple command line interface that provides access to all features. It can also be used as a library
via the fully documented API.


CLI
~~~

.. code-block:: text

    $ qecsim --version
    qecsim, version 1.0b1
    $ qecsim --help
    ...


API
~~~

.. code-block:: text

    >>> import qecsim
    >>> qecsim.__version__
    '1.0b1'
    >>> from qecsim import app
    >>> help(app)
    ...


Extension
~~~~~~~~~

See https://bitbucket.org/qecsim/qecsimext for a basic example demonstrating how qecsim can be extended with additional
codes, error models and decoders that integrate into the command-line interface.


Links
-----

* Source code: https://bitbucket.org/qecsim/qecsim
* Documentation: http://davidtuckett.com/qit/qecsim/
* Issue tracker: TODO
* Releases: TODO

----

Copyright 2016, David Tuckett.
