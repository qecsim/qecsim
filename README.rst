qecsim
======

.. image:: https://img.shields.io/badge/docs-stable-blue.svg
    :target: https://qecsim.github.io/
    :alt: Documentation

.. image:: https://github.com/qecsim/qecsim/workflows/CI/badge.svg?branch=master
    :target: https://github.com/qecsim/qecsim/actions?workflow=CI
    :alt: CI Status

.. image:: https://codecov.io/gh/qecsim/qecsim/branch/master/graph/badge.svg?token=BH80LCIYBZ
    :target: https://codecov.io/gh/qecsim/qecsim
    :alt: Coverage

.. image:: https://img.shields.io/pypi/v/qecsim
    :target: https://pypi.org/project/qecsim/
    :alt: PyPI

**qecsim** is a Python 3 package for simulating quantum error correction using
stabilizer codes.

It provides access to all features via a command-line interface. It can also be
used as a library via the fully-documented API. It includes many common codes,
error models and decoders, and can be extended with additional components.


Installation
------------

Install and upgrade using `pip`_:

.. code-block:: text

    $ pip install -U qecsim

.. _pip: https://pip.pypa.io/en/stable/quickstart/


Usage
-----


CLI
~~~

.. code-block:: text

    $ qecsim --version
    qecsim, version 1.0b9
    $ qecsim --help                 # console script
    ...
    $ python -O -m qecsim --help    # module script with Python options e.g. -O for optimize
    ...


API
~~~

.. code-block:: text

    >>> import qecsim
    >>> qecsim.__version__
    '1.0b9'
    >>> from qecsim import app
    >>> help(app)
    ...


Extension
~~~~~~~~~

qecsim can be extended with additional codes, error models and decoders that
integrate into the command-line interface.
See https://github.com/qecsim/qecsimext for a basic example.


Demo Notebooks
--------------

The ``demos/`` directory contains Jupyter notebooks showing various QEC simulations:

- ``demo_basic.ipynb`` — Basic QEC simulation
- ``demo_basic_plot.ipynb`` — Basic simulation with plots
- ``demo_planar.ipynb`` — Planar code simulation
- ``demo_planar_plot.ipynb`` — Planar code with plots
- ``demo_toric.ipynb`` — Toric code simulation
- ``demo_toric_plot.ipynb`` — Toric code with plots
- ``demo_rotatedplanar.ipynb`` — Rotated planar code
- ``demo_rotatedplanar_plot.ipynb`` — Rotated planar with plots
- ``demo_color666.ipynb`` — Color code (6.6.6)
- ``demo_color666_plot.ipynb`` — Color code with plots
- ``demo_planar_correlated.ipynb`` — Planar code with correlated errors
- ``qsu.ipynb`` — Quantum surface unit demo
- ``demo_code_comparison.ipynb`` — **NEW**: Compare Surface vs Toric codes (phase diagrams, overhead, error suppression)


Fork Notes
----------

This fork (by @simbot912) includes:

- Fix: demo notebooks now included in pip installs (MANIFEST.in update)
- Added: ``demo_code_comparison.ipynb`` — side-by-side comparison of surface and toric codes


License / Citing
----------------

qecsim is released under the BSD 3-Clause license. If you use qecsim in your
research, please see the `qecsim documentation`_ for citing details.

.. _qecsim documentation: https://qecsim.github.io/


Links
-----

* Source code: https://github.com/qecsim/qecsim
* Documentation: https://qecsim.github.io/
* Issue tracker: https://github.com/qecsim/qecsim/issues
* Releases: https://pypi.org/project/qecsim/
* Contact: qecsim@gmail.com

----

Copyright 2016 - 2021, David K. Tuckett.
