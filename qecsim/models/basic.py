"""
This module contains implementations relevant to basic stabilizer codes.
"""

import functools

from qecsim import paulitools as pt
from qecsim.model import StabilizerCode


class BasicCode(StabilizerCode):
    """Implements a basic code defined by its stabilizers and logical operators."""

    def __init__(self, pauli_stabilizers, pauli_logical_xs, pauli_logical_zs, n_k_d=None, label=None):
        """
        Initialise new basic code.

        Assumptions:

        * Paulis are expressed as a string of capitalized I, X, Y, Z characters.
        * Paulis are all the same length, with one character for each physical qubit.
        * The logical X and Z operators are in matching order, with one of each for each logical qubit.

        :param pauli_stabilizers: Pauli stabilizers.
        :type pauli_stabilizers: tuple of str
        :param pauli_logical_xs: Pauli logical X operators.
        :type pauli_logical_xs: tuple of str
        :param pauli_logical_zs: Pauli logical Z operators.
        :type pauli_logical_zs: tuple of str
        :param n_k_d: Descriptor in the format (n, k, d). (Optional. Defaults to n and k calculated and d None.)
        :type n_k_d: 3-tuple of int
        :param label: Label suitable for use in plots. (Optional. Defaults to 'Basic [n, k, d]'.)
        :type label: str
        """
        self._pauli_stabilizers = pauli_stabilizers
        self._pauli_logical_xs = pauli_logical_xs
        self._pauli_logical_zs = pauli_logical_zs
        self._n_k_d = (
            len(self._pauli_stabilizers[0]) if self._pauli_stabilizers else 0,
            len(self._pauli_logical_xs) if self._pauli_logical_xs else 0,
            None
        ) if n_k_d is None else n_k_d
        self._label = 'Basic [{},{},{}]'.format(*self.n_k_d) if label is None else label

    @property
    @functools.lru_cache()
    def stabilizers(self):
        """See :meth:`qecsim.model.StabilizerCode.stabilizers`"""
        return pt.pauli_to_bsf(self._pauli_stabilizers)

    @property
    @functools.lru_cache()
    def logical_xs(self):
        """See :meth:`qecsim.model.StabilizerCode.logical_xs`"""
        return pt.pauli_to_bsf(self._pauli_logical_xs)

    @property
    @functools.lru_cache()
    def logical_zs(self):
        """See :meth:`qecsim.model.StabilizerCode.logical_zs`"""
        return pt.pauli_to_bsf(self._pauli_logical_zs)

    @property
    def n_k_d(self):
        """See :meth:`qecsim.model.StabilizerCode.n_k_d`"""
        return self._n_k_d

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return self._label

    def __eq__(self, other):
        if isinstance(other, BasicCode):
            return ((self._pauli_stabilizers, self._pauli_logical_xs, self._pauli_logical_zs, self._n_k_d,
                     self._label) ==
                    (other._pauli_stabilizers, other._pauli_logical_xs, other._pauli_logical_zs, other._n_k_d,
                     other._label))
        return NotImplemented

    def __hash__(self):
        return hash((self._pauli_stabilizers, self._pauli_logical_xs, self._pauli_logical_zs, self._n_k_d, self._label))

    def __repr__(self):
        return '{}({!r}, {!r}, {!r}, {!r}, {!r})'.format(
            type(self).__name__,
            self._pauli_stabilizers,
            self._pauli_logical_xs,
            self._pauli_logical_zs,
            self.n_k_d,
            self.label,
        )


class FiveQubitCode(BasicCode):
    """Implements the 5-qubit [5, 1, 3] code."""

    def __init__(self):
        """Initialise new 5-qubit [5, 1, 3] code."""
        super().__init__(
            ('XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'),
            ('XXXXX',),
            ('ZZZZZ',),
            (5, 1, 3),
            '5-qubit'
        )

    def __repr__(self):
        return '{}()'.format(type(self).__name__)


class SteaneCode(BasicCode):
    """Implements the Steane [7, 1, 3] code."""

    def __init__(self):
        """Initialise new Steane [7, 1, 3] code."""
        super().__init__(
            ('IIIXXXX', 'IXXIIXX', 'XIXIXIX', 'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ'),
            ('XXXXXXX',),
            ('ZZZZZZZ',),
            (7, 1, 3),
            'Steane'
        )

    def __repr__(self):
        return '{}()'.format(type(self).__name__)
