r"""
Introduction
------------

qecsim is a Python 3 package for simulating quantum error correction using stabilizer codes.

It is lightweight, modular and extensible, allowing additional codes, error models and decoders to be plugged in.

Components
----------

qecsim includes three key abstract classes: :class:`qecsim.model.StabilizerCode`, :class:`qecsim.model.ErrorModel` and
:class:`qecsim.model.Decoder`.

+----------------+
| StabilizerCode |
+================+
| | n_k_d        |
| | stabilizers  |
| | logical_xs   |
| | logical_zs   |
| | logicals     |
| |              |
| | validate()   |
+----------------+

+-----------------------------------------------+
| ErrorModel                                    |
+===============================================+
| | probability_distribution(probability):tuple |
| | generate(code, probability):error           |
+-----------------------------------------------+

+-----------------------------+
| Decoder                     |
+=============================+
| | decode(syndrome):recovery |
+-----------------------------+

A simulation run is executed by passing implementations of the key classes, along with an ``error_probability`` to the
function :func:`qecsim.app.run_once`.

+----------------------------------------------------------------------------------------------------+
| app                                                                                                |
+====================================================================================================+
| | run_once(code, error_model, decoder, error_probability):run_data                                 |
| |                                                                                                  |
| |  1. :math:`S \leftarrow` code.stabilizers                                                        |
| |  2. :math:`L \leftarrow` code.logicals                                                           |
| |  3. :math:`e \leftarrow` error_model.generate(code, error_probability)                           |
| |  4. :math:`y \leftarrow e \odot S^T`                                                             |
| |  5. :math:`r \leftarrow` decoder.decode(code, :math:`y`)                                         |
| |  6. sanity check :math:`r \odot S^T = y`                                                         |
| |  7. success :math:`\iff (r \oplus e) \odot L^T = 0`                                              |
| |                                                                                                  |
| | run(code, error_model, decoder, error_probabilities, max_runs, max_failures):runs_data           |
| |                                                                                                  |
| | merge(runs_data_list, ...):runs_data_list                                                        |
+----------------------------------------------------------------------------------------------------+

Notes
~~~~~

-   The binary symplectic product :math:`\odot` is defined as :math:`A \odot B \equiv A \Lambda B \bmod 2` where
    :math:`\Lambda = \left[\begin{matrix} 0 & I \\ I & 0 \end{matrix}\right]`.

-   Pauli operators - ``stabilizers``, ``logicals``, ``error``, ``recovery`` - are represented in binary symplectic form
    by NumPy arrays::

        numpy.array([0, 0, 1, 1, 0, 1, 1, 0])  # [0 0 1 1 | 0 1 1 0] = IZYX

-   The :mod:`qecsim.app` module also includes functions to execute fault-tolerant simulations, see
    :func:`qecsim.app.run_once_ftp` and :func:`qecsim.app.run_ftp`, which delegate to a fault-tolerant decoder
    implementation of :class:`qecsim.model.DecoderFTP`.

-   The :mod:`qecsim.paulitools` module provides utility functions for manipulating Pauli operators in string and binary
    symplectic form.

-   The :mod:`qecsim.graphtools` and :mod:`qecsim.tensortools` modules provide support for graph matching and tensor
    network contraction as used by some decoder implementations.

Implementations
---------------

Stablizer code implementations include:

-   5-qubit code :class:`qecsim.models.basic.FiveQubitCode`
-   7-qubit Steane code :class:`qecsim.models.basic.SteaneCode`
-   Color 6.6.6 code :class:`qecsim.models.color.Color666Code`
-   Planar code :class:`qecsim.models.planar.PlanarCode`
-   Rotated planar code :class:`qecsim.models.rotatedplanar.RotatedPlanarCode`
-   Rotated toric code :class:`qecsim.models.rotatedtoric.RotatedToricCode`
-   Toric code :class:`qecsim.models.toric.ToricCode`

Error model implementations include:

-   Biased-depolarizing error model :class:`qecsim.models.generic.BiasedDepolarizingErrorModel`
-   Biased-Y-X error model :class:`qecsim.models.generic.BiasedYXErrorModel`
-   Bit-flip error model :class:`qecsim.models.generic.BitFlipErrorModel`
-   Bit-phase-flip error model :class:`qecsim.models.generic.BitPhaseFlipErrorModel`
-   Center slice error model :class:`qecsim.models.generic.CenterSliceErrorModel`
-   Depolarizing error model :class:`qecsim.models.generic.DepolarizingErrorModel`
-   File error model :class:`qecsim.models.generic.FileErrorModel`
-   Phase-flip error model :class:`qecsim.models.generic.PhaseFlipErrorModel`

Decoder implementations include:

-   Color 6.6.6 MPS decoder :class:`qecsim.models.color.Color666MPSDecoder`
-   Naive decoder :class:`qecsim.models.generic.NaiveDecoder`
-   Planar Converging MWPM decoder :class:`qecsim.models.planar.PlanarCMWPMDecoder`
-   Planar MPS decoder :class:`qecsim.models.planar.PlanarMPSDecoder`
-   Planar MWPM decoder :class:`qecsim.models.planar.PlanarMWPMDecoder`
-   Planar Rotated MPS decoder :class:`qecsim.models.planar.PlanarRMPSDecoder`
-   Planar Y-noise decoder :class:`qecsim.models.planar.PlanarYDecoder`
-   Rotated planar MPS decoder :class:`qecsim.models.rotatedplanar.RotatedPlanarMPSDecoder`
-   Rotated planar Rotated MPS decoder :class:`qecsim.models.rotatedplanar.RotatedPlanarRMPSDecoder`
-   Rotated planar Symmetry MWPM decoder :class:`qecsim.models.rotatedplanar.RotatedPlanarSMWPMDecoder`
-   Rotated toric Symmetry MWPM decoder :class:`qecsim.models.rotatedtoric.RotatedToricSMWPMDecoder`
-   Toric MWPM decoder :class:`qecsim.models.toric.ToricMWPMDecoder`

Further implementations can be added by extending the key abstract models.
"""

__version__ = '1.0b2'
