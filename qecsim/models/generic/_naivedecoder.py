import operator

import numpy as np

from qecsim import paulitools as pt
from qecsim.model import Decoder


class NaiveDecoder(Decoder):
    """
    Implements a naive decoder.

    Decoding algorithm:

    * Naively iterate through all possible errors, in ascending weight, and resolve to the first error that matches
      the syndrome.

    Notes:

    * Slow for large numbers of qubits and high weights.
    """

    _cli_help = """Naive ([max_qubits] INT)"""

    MAX_QUBITS = 10

    def __init__(self, max_qubits=MAX_QUBITS):
        """
        Initialise new naive decoder.

        Notes:

        * As this decoder is slow for large number of qubits, it is restricted by default. Set max_qubits to override.

        :param max_qubits: Maximum supported number of physical qubits. (default=10, unrestricted=falsy)
        :type max_qubits: int or None
        :raises ValueError: if max_qubits is not falsy or > 0.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if not (not max_qubits or operator.index(max_qubits) > 0):
                raise ValueError("NaiveDecoder valid max_qubits values are falsy or integer > 0")
        except TypeError as ex:
            raise TypeError('NaiveDecoder invalid parameter type') from ex
        self._max_qubits = max_qubits

    def decode(self, code, syndrome, **kwargs):
        """
        See :meth:`qecsim.model.Decoder.decode`

        :raises ValueError: if qubits in code exceeds max_qubits.
        """
        n_qubits = code.n_k_d[0]
        if self._max_qubits and n_qubits > self._max_qubits:
            raise ValueError('NaiveDecoder limited to {} qubits. {} code has {} qubits. '
                             'Set max_qubits to override limit.'.format(self._max_qubits, code.label, n_qubits))
        for error in pt.ibsf(n_qubits):
            if np.array_equal(pt.bsp(error, code.stabilizers.T), syndrome):
                return error

    @property
    def label(self):
        """See :meth:`qecsim.model.Decoder.label`"""
        return 'Naive'

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self._max_qubits)
