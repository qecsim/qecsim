qecsim
======

**qecsim** is a Python 3 package for simulating quantum error correction using
stabilizer codes.

It provides access to all features via a command-line interface. It can also be
used as a library via the fully-documented API. It includes many common codes,
error models and decoders, and can be extended with additional components.

Installation
------------

Install and upgrade using `pip`_:

.. code-block:: text

    $ pip install -U qecsim   # TODO: COMING SOON

.. _pip: https://pip.pypa.io/en/stable/quickstart/


Usage
-----

CLI
~~~

.. code-block:: text

    $ qecsim --version
    qecsim, version 1.0b2
    $ qecsim --help
    ...


API
~~~

.. code-block:: text

    >>> import qecsim
    >>> qecsim.__version__
    '1.0b2'
    >>> from qecsim import app
    >>> help(app)
    ...


Extension
~~~~~~~~~

qecsim can be extended with additional codes, error models and decoders that
integrate into the command-line interface.
See https://bitbucket.org/qecsim/qecsimext/ for a basic example.


Links
-----

* Source code: https://bitbucket.org/qecsim/qecsim/
* Documentation: https://davidtuckett.com/qit/qecsim/
* Issue tracker: https://bitbucket.org/qecsim/qecsim/issues
* Releases: TODO: COMING SOON
* Contact: qecsim@gmail.com

----

Copyright 2016, David Tuckett.
