qecsim demos
============

Comparing planar MPS and MWPM decoders for a correlated error
-------------------------------------------------------------

This demo shows verbosely that the matrix product state (MPS) decoder
can successfully recover from a correlated error on the planar code when
the minimum weight perfect matching (MWPM) decoder fails.

| For normal use, the simulation of a single error correction run is
  encapsulated in the function:
| ``qecsim.app.run_once(code, error_model, decoder, error_probability)``,
| and the simulation of many error correction runs is encapsulated in
  the function:
| ``qecsim.app.run(code, error_model, decoder, error_probability, max_runs, max_failures)``.

Initialise the models
~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %run qsu.ipynb  # color-printing functions
    from qecsim import paulitools as pt
    from qecsim.models.planar import PlanarCode, PlanarMPSDecoder, PlanarMWPMDecoder
    
    # initialise models
    my_code = PlanarCode(3, 3)
    my_mps_decoder = PlanarMPSDecoder()
    my_mwpm_decoder = PlanarMWPMDecoder()
    # print models
    print(my_code)
    print(my_mps_decoder)
    print(my_mwpm_decoder)


.. parsed-literal::

    PlanarCode(3, 3)
    PlanarMPSDecoder(None, 'c', None, None)
    PlanarMWPMDecoder()


Create a correlated error
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # error: correlated error
    error = my_code.new_pauli().site('Y', (2, 0), (2, 4)).to_bsf()
    qsu.print_pauli('error:\n{}'.format(my_code.new_pauli(error)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">error:
    ·─┬─·─┬─·
      ·   ·  
    <span style="color:magenta; font-weight:bold">Y</span>─┼─·─┼─<span style="color:magenta; font-weight:bold">Y</span>
      ·   ·  
    ·─┴─·─┴─·</pre></div>


Evaluate the syndrome
~~~~~~~~~~~~~~~~~~~~~

The syndrome is a binary array indicating the stabilizers with which the
error does not commute.

.. code:: ipython3

    # syndrome: stabilizers that do not commute with the error
    syndrome = pt.bsp(error, my_code.stabilizers.T)
    qsu.print_pauli('syndrome:\n{}'.format(my_code.ascii_art(syndrome)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">syndrome:
    ──┬───┬──
    <span style="color:blue; font-weight:bold">Z</span> │   │ <span style="color:blue; font-weight:bold">Z</span>
    ──<span style="color:red; font-weight:bold">X</span>───<span style="color:red; font-weight:bold">X</span>──
    <span style="color:blue; font-weight:bold">Z</span> │   │ <span style="color:blue; font-weight:bold">Z</span>
    ──┴───┴──</pre></div>


Decoding fails using the MWPM decoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, the recovery operation is found by a minimum weight
perfect matching (MWPM) decoder that processes X errors and Z errors
separately and so fails to find a successful recovery operation.

.. code:: ipython3

    # recovery: best match recovery operation based on decoder
    mwpm_recovery = my_mwpm_decoder.decode(my_code, syndrome)
    qsu.print_pauli('mwpm_recovery:\n{}'.format(my_code.new_pauli(mwpm_recovery)))
    qsu.print_pauli('mwpm_recovery ^ error:\n{}'.format(my_code.new_pauli(mwpm_recovery ^ error)))
    print('check mwpm_recovery ^ error commutes with stabilizers (i.e. all zeros by construction):\n{}\n'.format(
            pt.bsp(mwpm_recovery ^ error, my_code.stabilizers.T)))
    print('success iff mwpm_recovery ^ error commutes with logicals (i.e. all zeros):\n{}\n'.format(
            pt.bsp(mwpm_recovery ^ error, my_code.logicals.T)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">mwpm_recovery:
    ·─┬─·─┬─·
      ·   ·  
    <span style="color:red; font-weight:bold">X</span>─┼─<span style="color:blue; font-weight:bold">Z</span>─┼─<span style="color:red; font-weight:bold">X</span>
      ·   ·  
    ·─┴─·─┴─·</pre></div>



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">mwpm_recovery ^ error:
    ·─┬─·─┬─·
      ·   ·  
    <span style="color:blue; font-weight:bold">Z</span>─┼─<span style="color:blue; font-weight:bold">Z</span>─┼─<span style="color:blue; font-weight:bold">Z</span>
      ·   ·  
    ·─┴─·─┴─·</pre></div>


.. parsed-literal::

    check mwpm_recovery ^ error commutes with stabilizers (i.e. all zeros by construction):
    [0 0 0 0 0 0 0 0 0 0 0 0]
    
    success iff mwpm_recovery ^ error commutes with logicals (i.e. all zeros):
    [1 0]
    


Decoding succeeds using the MPS decoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case, the recovery operation is found by a matrix product state
(MPS) decoder that approximates a maximum likelihood decoder and so
succeeds in finding a successful recovery operation.

.. code:: ipython3

    # recovery: best match recovery operation based on decoder
    mps_recovery = my_mps_decoder.decode(my_code, syndrome)
    qsu.print_pauli('mps_recovery:\n{}'.format(my_code.new_pauli(mps_recovery)))
    qsu.print_pauli('mps_recovery ^ error:\n{}'.format(my_code.new_pauli(mps_recovery ^ error)))
    print('check mps_recovery ^ error commutes with stabilizers (i.e. all zeros by construction):\n{}\n'.format(
            pt.bsp(mps_recovery ^ error, my_code.stabilizers.T)))
    print('success iff mps_recovery ^ error commutes with logicals (i.e. all zeros):\n{}\n'.format(
            pt.bsp(mps_recovery ^ error, my_code.logicals.T)))



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">mps_recovery:
    <span style="color:red; font-weight:bold">X</span>─┬─·─┬─<span style="color:red; font-weight:bold">X</span>
      ·   ·  
    <span style="color:blue; font-weight:bold">Z</span>─┼─·─┼─<span style="color:blue; font-weight:bold">Z</span>
      ·   ·  
    <span style="color:red; font-weight:bold">X</span>─┴─·─┴─<span style="color:red; font-weight:bold">X</span></pre></div>



.. raw:: html

    <div class="highlight"><pre style="line-height:1!important;">mps_recovery ^ error:
    <span style="color:red; font-weight:bold">X</span>─┬─·─┬─<span style="color:red; font-weight:bold">X</span>
      ·   ·  
    <span style="color:red; font-weight:bold">X</span>─┼─·─┼─<span style="color:red; font-weight:bold">X</span>
      ·   ·  
    <span style="color:red; font-weight:bold">X</span>─┴─·─┴─<span style="color:red; font-weight:bold">X</span></pre></div>


.. parsed-literal::

    check mps_recovery ^ error commutes with stabilizers (i.e. all zeros by construction):
    [0 0 0 0 0 0 0 0 0 0 0 0]
    
    success iff mps_recovery ^ error commutes with logicals (i.e. all zeros):
    [0 0]
    

