qecsim demos
============

Simulating error correction with a rotated planar stabilizer code
-----------------------------------------------------------------

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
    from qecsim.models.generic import DepolarizingErrorModel
    from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder
    
    # initialise models
    my_code = RotatedPlanarCode(7, 7)
    my_error_model = DepolarizingErrorModel()
    my_decoder = RotatedPlanarMPSDecoder(chi=8)
    # print models
    print(my_code)
    print(my_error_model)
    print(my_decoder)


.. parsed-literal::

    RotatedPlanarCode(7, 7)
    DepolarizingErrorModel()
    RotatedPlanarMPSDecoder(8, 'c', None)


Generate a random error
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # set physical error probability to 10%
    error_probability = 0.1
    # seed random number generator for repeatability
    rng = np.random.default_rng(10)
    
    # error: random error based on error probability
    error = my_error_model.generate(my_code, error_probability, rng)
    qsu.print_pauli('error:\n{}'.format(my_code.new_pauli(error)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">error:
      ┌───┐   ┌───┐   ┌───┐      
      ·───·───·───·───·───·───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─·───·───·───·───·───·───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─·───·───·───·───·───·───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─·───·───<span style="color:blue; font-weight:bold">Z</span>───·───·───·───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─·───<span style="color:magenta; font-weight:bold">Y</span>───·───·───·───<span style="color:red; font-weight:bold">X</span>───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─·───·───<span style="color:magenta; font-weight:bold">Y</span>───·───·───·───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─<span style="color:magenta; font-weight:bold">Y</span>───·───·───·───·───·───·  
          └───┘   └───┘   └───┘  </pre></div>


Evaluate the syndrome
~~~~~~~~~~~~~~~~~~~~~

The syndrome is a binary array indicating the stabilizers with which the
error does not commute. It is calculated as
:math:`syndrome = error \odot stabilisers^T`.

.. code:: ipython3

    # syndrome: stabilizers that do not commute with the error
    syndrome = pt.bsp(error, my_code.stabilizers.T)
    qsu.print_pauli('syndrome:\n{}'.format(my_code.ascii_art(syndrome)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">syndrome:
      ┌───┐   ┌───┐   ┌───┐      
      ·───·───·───·───·───·───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─·───·───·───·───·───·───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─·───·───·───·───·───·───·─┐
      │░░░│   │░<span style="color:red; font-weight:bold">X</span>░│   │░░░│   │░│
    ┌─·───·───·───·───·───·───·─┘
    │░│ <span style="color:blue; font-weight:bold">Z</span> │░░░│   │░░░│ <span style="color:blue; font-weight:bold">Z</span> │░░░│  
    └─·───·───·───·───·───·───·─┐
      │░<span style="color:red; font-weight:bold">X</span>░│   │░<span style="color:red; font-weight:bold">X</span>░│   │░░░│ <span style="color:blue; font-weight:bold">Z</span> │░│
    ┌─·───·───·───·───·───·───·─┘
    <span style="color:red; font-weight:bold">X</span>░│ <span style="color:blue; font-weight:bold">Z</span> │░<span style="color:red; font-weight:bold">X</span>░│ <span style="color:blue; font-weight:bold">Z</span> │░░░│   │░░░│  
    └─·───·───·───·───·───·───·  
          └───┘   └───┘   └───┘  </pre></div>


Find a recovery operation
~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, the recovery operation is found by contracting a tensor
network defined to have the value of the coset :

-  Create a sample recovery operation by applying strings of Paulis
   between syndrome plaquettes and appropriate boundaries.
-  Define tensor networks corresponding to the probability of each left
   coset of the generator group with the sample recovery and logical
   Pauli operations.
-  Contract each tensor network (approximately) to evaluate the coset
   probabilities.
-  Return any recovery from the most probable coset.

.. code:: ipython3

    # recovery: best match recovery operation based on decoder
    recovery = my_decoder.decode(my_code, syndrome)
    qsu.print_pauli('recovery:\n{}'.format(my_code.new_pauli(recovery)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">recovery:
      ┌───┐   ┌───┐   ┌───┐      
      ·───·───·───·───·───·───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─·───·───·───·───·───·───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─·───·───·───·───·───·───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─·───·───<span style="color:blue; font-weight:bold">Z</span>───·───·───·───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─·───<span style="color:red; font-weight:bold">X</span>───<span style="color:magenta; font-weight:bold">Y</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>───·───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─<span style="color:magenta; font-weight:bold">Y</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─·───<span style="color:magenta; font-weight:bold">Y</span>───<span style="color:red; font-weight:bold">X</span>───·───·───·───·  
          └───┘   └───┘   └───┘  </pre></div>


As a sanity check, we expect :math:`recovery \oplus error` to commute
with all stabilizers,
i.e. :math:`(recovery \oplus error) \odot stabilisers^T = 0`.

.. code:: ipython3

    # check recovery ^ error commutes with stabilizers (by construction)
    print(pt.bsp(recovery ^ error, my_code.stabilizers.T))


.. parsed-literal::

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0]


Visualise :math:`recovery \oplus error`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just out of curiosity, we can see what :math:`recovery \oplus error`
looks like. If successful, it should be a product of stabilizer
plaquette operators.

.. code:: ipython3

    # print recovery ^ error (out of curiosity)
    qsu.print_pauli('recovery ^ error:\n{}'.format(my_code.new_pauli(recovery ^ error)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">recovery ^ error:
      ┌───┐   ┌───┐   ┌───┐      
      ·───·───·───·───·───·───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─·───·───·───·───·───·───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─·───·───·───·───·───·───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─·───·───·───·───·───·───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─·───<span style="color:blue; font-weight:bold">Z</span>───<span style="color:magenta; font-weight:bold">Y</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>───·─┐
      │░░░│   │░░░│   │░░░│   │░│
    ┌─<span style="color:magenta; font-weight:bold">Y</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:blue; font-weight:bold">Z</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>───·─┘
    │░│   │░░░│   │░░░│   │░░░│  
    └─<span style="color:magenta; font-weight:bold">Y</span>───<span style="color:magenta; font-weight:bold">Y</span>───<span style="color:red; font-weight:bold">X</span>───·───·───·───·  
          └───┘   └───┘   └───┘  </pre></div>


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
operation. The rotated planar 7 x 7 code has distance :math:`d = 7` so
we can only guarantee to correct errors up to weight
:math:`(d - 1)/2=3`.

Equivalent code in single call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above demo is equivalent to the following code.

.. code:: ipython3

    # repeat demo in single call
    from qecsim import app
    print(app.run_once(my_code, my_error_model, my_decoder, error_probability))


.. parsed-literal::

    {'error_weight': 4, 'success': True}

