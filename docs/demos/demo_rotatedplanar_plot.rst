qecsim demos
============

Plotting logical failure rate as a function of physical error rate for rotated planar stabilizer codes
------------------------------------------------------------------------------------------------------

This demo shows how to plot statistics on many error correction
simulation runs, see figure for more details.

The code makes use of
``qecsim.app.run(code, error_model, decoder, error_probability, max_runs, max_failures)``.

Initialise run parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %matplotlib inline
    import collections
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from qecsim import app
    from qecsim.models.generic import BitPhaseFlipErrorModel
    from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder
    
    # set models
    codes = [RotatedPlanarCode(*size) for size in [(3, 3), (5, 5), (7, 7), (9, 9)]]
    error_model = BitPhaseFlipErrorModel()
    decoder = RotatedPlanarMPSDecoder(chi=8)
    # set physical error probabilities
    error_probability_min, error_probability_max = 0, 0.5
    error_probabilities = np.linspace(error_probability_min, error_probability_max, 20)
    # set max_runs for each probability
    max_runs = 10000
    
    # print run parameters
    print('Codes:', [code.label for code in codes])
    print('Error model:', error_model.label)
    print('Decoder:', decoder.label)
    print('Error probabilities:', error_probabilities)
    print('Maximum runs:', max_runs)


.. parsed-literal::

    Codes: ['Rotated planar 3x3', 'Rotated planar 5x5', 'Rotated planar 7x7', 'Rotated planar 9x9']
    Error model: Bit-phase-flip
    Decoder: Rotated planar MPS (chi=8, mode=c)
    Error probabilities: [0.         0.02631579 0.05263158 0.07894737 0.10526316 0.13157895
     0.15789474 0.18421053 0.21052632 0.23684211 0.26315789 0.28947368
     0.31578947 0.34210526 0.36842105 0.39473684 0.42105263 0.44736842
     0.47368421 0.5       ]
    Maximum runs: 10000


Run simulations
~~~~~~~~~~~~~~~

.. code:: ipython3

    # run simulations and print data from middle run to view format
    data = [app.run(code, error_model, decoder, error_probability, max_runs=max_runs)
            for code in codes for error_probability in error_probabilities]
    print(data[len(data)//2])


.. parsed-literal::

    {'code': 'Rotated planar 7x7', 'n_k_d': (49, 1, 7), 'time_steps': 1, 'error_model': 'Bit-phase-flip', 'decoder': 'Rotated planar MPS (chi=8, mode=c)', 'error_probability': 0.0, 'measurement_error_probability': 0.0, 'n_run': 10000, 'n_success': 10000, 'n_fail': 0, 'error_weight_total': 0, 'error_weight_pvar': 0, 'logical_failure_rate': 0.0, 'physical_error_rate': 0.0, 'wall_time': 623.0377811500002}


Prepare data for plotting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # prepare code to x,y map and print
    code_to_xys = {}
    for run in data:
        xys = code_to_xys.setdefault(run['code'], [])
        xys.append((run['physical_error_rate'], run['logical_failure_rate']))
    print('\n'.join('{}: {}'.format(k, v) for k, v in code_to_xys.items()))


.. parsed-literal::

    Rotated planar 3x3: [(0.0, 0.0), (0.02652222222222222, 0.0), (0.05328888888888889, 0.0), (0.07881111111111111, 0.0005), (0.10457777777777778, 0.001), (0.13224444444444444, 0.0026), (0.1578, 0.0068), (0.18645555555555557, 0.0147), (0.20952222222222222, 0.0229), (0.23814444444444444, 0.0393), (0.2620444444444444, 0.0556), (0.2882777777777778, 0.0871), (0.3180555555555556, 0.1255), (0.3412777777777778, 0.1578), (0.3663888888888889, 0.1978), (0.39576666666666666, 0.2538), (0.4221333333333333, 0.3199), (0.4466666666666667, 0.37), (0.4727, 0.4326), (0.4997888888888889, 0.4973)]
    Rotated planar 5x5: [(0.0, 0.0), (0.025936, 0.0), (0.052516, 0.0), (0.079112, 0.0), (0.10588399999999999, 0.0), (0.131896, 0.0), (0.15755999999999998, 0.0002), (0.183908, 0.0002), (0.210472, 0.0004), (0.235948, 0.0013), (0.261804, 0.0056), (0.29042399999999996, 0.0124), (0.31702800000000003, 0.0278), (0.34118000000000004, 0.0484), (0.369148, 0.0889), (0.393752, 0.1425), (0.41991999999999996, 0.2087), (0.44654799999999994, 0.2922), (0.47332399999999997, 0.3963), (0.500432, 0.5025)]
    Rotated planar 7x7: [(0.0, 0.0), (0.026026530612244896, 0.0), (0.05289591836734694, 0.0), (0.07898163265306123, 0.0), (0.10591020408163265, 0.0), (0.1312795918367347, 0.0), (0.15808979591836736, 0.0), (0.18390204081632652, 0.0), (0.20982857142857142, 0.0), (0.23698775510204081, 0.0001), (0.26319387755102036, 0.0004), (0.2891591836734694, 0.001), (0.3147204081632653, 0.0037), (0.3422734693877551, 0.0096), (0.36848571428571425, 0.029), (0.3953061224489796, 0.0687), (0.4207714285714285, 0.1319), (0.447238775510204, 0.2282), (0.47406122448979593, 0.3609), (0.49954897959183675, 0.5006)]
    Rotated planar 9x9: [(0.0, 0.0), (0.026504938271604935, 0.0), (0.052658024691358024, 0.0), (0.07902962962962963, 0.0), (0.10498765432098764, 0.0), (0.13198395061728396, 0.0), (0.15845061728395063, 0.0), (0.18364444444444444, 0.0), (0.2106864197530864, 0.0), (0.23753333333333335, 0.0), (0.26357901234567904, 0.0), (0.2892901234567901, 0.0), (0.31629753086419754, 0.0003), (0.3424358024691358, 0.0017), (0.36902345679012344, 0.0088), (0.3947320987654321, 0.0276), (0.42084691358024695, 0.0747), (0.44793456790123454, 0.1737), (0.47360740740740737, 0.3156), (0.49949135802469136, 0.5076)]


Plot logical failure rate against physical error rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    # format plot
    fig = plt.figure(1, figsize=(12, 9))
    plt.title('Rotated planar code simulation\n({} error model, {} decoder)'.format(error_model.label, decoder.label))
    plt.xlabel('Physical error rate')
    plt.ylabel('Logical failure rate')
    plt.xlim(error_probability_min-0.05, error_probability_max+0.05)
    plt.ylim(-0.05, 0.65)
    # add data
    for code, xys in code_to_xys.items():
        plt.plot(*zip(*xys), 'x-', label='{} code'.format(code))
    plt.legend(loc='upper left')
    plt.show()



.. image:: demo_rotatedplanar_plot_files/demo_rotatedplanar_plot_9_0.png


Fig. 1. Logical failure rate as a function of physical error rate for
rotated planar stabilizer codes.

Definitions:

-  | A physical error occurs when the generated error acts non-trivially
     on a physical qubit.
   | :math:`(\text{physical error rate}) = mean(\text{error weight}) / (\text{number of physical qubits})`.

-  | A logical failure occurs when the combined action of the generated
     error and recovery operation acts non-trivially on the codespace,
     irrespective of the number of logical qubits encoded.
   | :math:`(\text{logical failure rate}) = (\text{number of logical failures}) / (\text{number of runs})`.
