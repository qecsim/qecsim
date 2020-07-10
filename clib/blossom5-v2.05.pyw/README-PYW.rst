MINIMAL PYTHON CTYPES WRAPPER FOR BLOSSOM V
===========================================

This archive is a Python wrapper for `Blossom V`_, a fast C++ implementation of minimum-weight perfect matching in a
graph, due to Vladimir Kolmogorov:

    Vladimir Kolmogorov. "Blossom V: A new implementation of a minimum cost perfect matching algorithm."
    In *Mathematical Programming Computation* (MPC), July 2009, 1(1):43-67.

.. _Blossom V: http://pub.ist.ac.at/~vnk/software.html

The licence for Blossom V does not permit public redistribution of the code or its derivatives (see `Blossom V`_ and the
Blossom V README.TXT file for full details of the license and citation requirements). Therefore, Blossom V is not
packaged with this archive.

If your use case satisfies the license requirements of Blossom V, you can build a C++ library that makes Blossom V
accessible from Python as follows:

* Copy this Python wrapper ``blossom5-v2.05.pyw.tar.gz`` to a new directory.
* Download Blossom V ``blossom5-v2.05.src.tar.gz`` from http://pub.ist.ac.at/~vnk/software.html
* Build C++ library ``libpypm.so`` from Blossom V and the Python wrapper.

.. code-block:: text

    $ tar -xzf blossom5-v2.05.src.tar.gz
    $ tar -xzf blossom5-v2.05.pyw.tar.gz
    $ cp blossom5-v2.05.pyw/* blossom5-v2.05.src/
    $ cd blossom5-v2.05.src/
    $ make -f MakeFile-pyw
    ...
    c++ -shared ... -o libpypm.so

* Check that Blossom V is accessible from Python:

.. code-block:: text

    $ python
    >>> import pypm_example
    >>> pypm_example.INFTY
    1073741823
    >>>
    >>> edges = [(1, 2, 10), (1, 3, 25), (0, 2, 56), (0, 1, 15), (2, 3, 6)]
    >>> mates = pypm_example.mwpm_ids(edges)
    >>> mates
    {(0, 1), (2, 3)}
    >>>
    >>> edges = [('b', 'c', 10), ('b', 'd', 25), ('a', 'c', 56), ('a', 'b', 15), ('c', 'd', 6)]
    >>> mates = pypm_example.mwpm(edges)
    >>> mates
    {('c', 'd'), ('a', 'b')}
    >>>

* See ``pypm_example.py`` for further details on loading and using the C++ library ``libpypm.so``.

The above procedure has been tested on Linux and MacOS. The Blossom V README.TXT file states that it should compile with
the Microsoft Visual C++ compiler, therefore the above procedure, with some adaptation, *should* work on Windows.

----

This Python wrapper was developed to enable the use of Blossom V with qecsim: https://bitbucket.org/qecsim/qecsim/

Copyright 2016, David Tuckett.
