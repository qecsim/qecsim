qecsim demos
============

Simulating error correction with a basic stabilizer code
--------------------------------------------------------

This demo shows verbosely how to simulate one error correction run.

| For normal use, the code in this demo is encapsulated in the function:
| ``qecsim.app.run_once(code, error_model, decoder, error_probability)``,
| and the simulation of many error correction runs is encapsulated in
  the function:
| ``qecsim.app.run(code, error_model, decoder, error_probability, max_runs, max_failures)``.

Notes:

-  Operators can be visualised in binary symplectic form (bsf) or Pauli
   form, e.g. ``[1 1 0|0 1 0] = XYI``.
-  The binary symplectic product is denoted by :math:`\odot` and defined
   as :math:`A \odot B \equiv A \Lambda B \bmod 2` where
   :math:`\Lambda = \left[\begin{matrix} 0 & I \\ I & 0 \end{matrix}\right]`.
-  Binary addition is denoted by :math:`\oplus` and defined as addition
   modulo 2, or equivalently exclusive-or.

Initialise the models
~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %run qsu.ipynb  # color-printing functions
    import numpy as np
    from qecsim import paulitools as pt
    from qecsim.models.generic import DepolarizingErrorModel, NaiveDecoder
    from qecsim.models.basic import FiveQubitCode
    
    # initialise models
    my_code = FiveQubitCode()
    my_error_model = DepolarizingErrorModel()
    my_decoder = NaiveDecoder()
    # print models
    print(my_code)
    print(my_error_model)
    print(my_decoder)


.. parsed-literal::

    FiveQubitCode()
    DepolarizingErrorModel()
    NaiveDecoder(10)


Generate a random error
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # set physical error probability to 10%
    error_probability = 0.1
    # seed random number generator for repeatability
    rng = np.random.default_rng(8)
    
    # error: random error based on error probability
    error = my_error_model.generate(my_code, error_probability, rng)
    qsu.print_pauli('error: {} {}'.format(error, pt.bsf_to_pauli(error)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">error: [0 0 0 0 0 0 1 0 0 0] I<span style="color:blue; font-weight:bold">Z</span>III</pre></div>


Evaluate the syndrome
~~~~~~~~~~~~~~~~~~~~~

The syndrome is a binary array indicating the stabilizers with which the
error does not commute. It is calculated as
:math:`syndrome = error \odot stabilisers^T`.

.. code:: ipython3

    # syndrome: stabilizers that do not commute with the error
    syndrome = pt.bsp(error, my_code.stabilizers.T)
    print('syndrome: {}'.format(syndrome))


.. parsed-literal::

    syndrome: [0 1 0 1]


Find a recovery operation
~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, the recovery operation is found by using a naive decoder
that iterates through all possible Pauli operations, in ascending
weight, until it finds a recovery operation that gives the same syndrome
as the random error,
i.e. :math:`recovery \odot stabilisers^T = syndrome`.

.. code:: ipython3

    # recovery: best match recovery operation based on decoder
    recovery = my_decoder.decode(my_code, syndrome)
    qsu.print_pauli('recovery: {} {}'.format(recovery, pt.bsf_to_pauli(recovery)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">recovery: [0 0 0 0 0 0 1 0 0 0] I<span style="color:blue; font-weight:bold">Z</span>III</pre></div>


As a sanity check, we expect :math:`recovery \oplus error` to commute
with all stabilizers,
i.e. :math:`(recovery \oplus error) \odot stabilisers^T = 0`.

.. code:: ipython3

    # check recovery ^ error commutes with stabilizers (by construction)
    print(pt.bsp(recovery ^ error, my_code.stabilizers.T))


.. parsed-literal::

    [0 0 0 0]


Test if the recovery operation is successful
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recovery operation is successful iff :math:`recovery \oplus error`
commutes with all logical operators,
i.e. :math:`(recovery \oplus error) \odot logicals^T = 0.`

.. code:: ipython3

    # success iff recovery ^ error commutes with logicals
    print(pt.bsp(recovery ^ error, my_code.logicals.T))


.. parsed-literal::

    [0 0]


Note: The decoder is not guaranteed to find a successful recovery
operation. The five qubit code has distance :math:`d = 3` so we can only
guarantee to correct errors up to weight :math:`(d - 1)/2=1`.

Equivalent code in single call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above demo is equivalent to the following code.

.. code:: ipython3

    # repeat demo in single call
    from qecsim import app
    print(app.run_once(my_code, my_error_model, my_decoder, error_probability))


.. parsed-literal::

    {'error_weight': 1, 'success': True}

