import functools
import itertools
import operator

import numpy as np

from qecsim.model import StabilizerCode, cli_description
from qecsim.models.color import Color666Pauli


@cli_description('Color 6.6.6 (size INT odd >=3)')
class Color666Code(StabilizerCode):
    r"""
    Implements a triangular color 6.6.6 code.

    Indices:

    * Indices are in the format (row, column).

    For example, a size 5 triangular lattice with site indices in (parentheses) and plaquette indices in [brackets]:
    ::

        (0,0)
          |
          |
        (1,0)   [1,1]
             \
               \
        [2,0]   (2,1)---(2,2)
                  |          \
                  |            \
        (3,0)---(3,1)   [3,2]   (3,3)
          |          \            |
          |            \          |
        (4,0)   [4,1]   (4,2)---(4,3)   [4,4]
             \            |          \
               \          |            \
        [5,0]   (5,1)---(5,2)   [5,3]   (5,4)---(5,5)
                  |          \            |          \
                  |            \          |            \
        (6,0)---(6,1)   [6,2]   (6,3)---(6,4)   [6,5]   (6,6)

    """

    MIN_SIZE = 3

    def __init__(self, size):
        """
        Initialise new triangular color 6.6.6 code.

        :param size: Size of side of triangular lattice in terms of number of qubits.
        :type size: int
        :raises ValueError: if size smaller than 3.
        :raises ValueError: if size is even.
        :raises TypeError: if any parameter is of an invalid type.
        """
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if operator.index(size) < self.MIN_SIZE:
                raise ValueError('{} minimum size is {}.'.format(type(self).__name__, self.MIN_SIZE))
            if size % 2 == 0:
                raise ValueError('{} size must be odd.'.format(type(self).__name__))
        except TypeError as ex:
            raise TypeError('{} invalid parameter type'.format(type(self).__name__)) from ex
        self._size = size

    # < StabilizerCode interface methods >

    @property
    @functools.lru_cache()
    def n_k_d(self):
        """See :meth:`qecsim.model.StabilizerCode.n_k_d`"""
        # e.g. size=7: k=1, d=7,  n=7+6+5+5+4+3+3+2+1+1=(d+1)d/2+((d-1)/2)^2=(3d^2+1)4
        return (3 * self.size ** 2 + 1) // 4, 1, self.size

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return 'Color 6.6.6 {}'.format(self.size)

    @property
    @functools.lru_cache()
    def stabilizers(self):
        """See :meth:`qecsim.model.StabilizerCode.stabilizers`"""
        return np.array([self.new_pauli().plaquette('X', i).to_bsf() for i in self._plaquette_indices]
                        + [self.new_pauli().plaquette('Z', i).to_bsf() for i in self._plaquette_indices])

    @property
    @functools.lru_cache()
    def logical_xs(self):
        """See :meth:`qecsim.model.StabilizerCode.logical_xs`"""
        return np.array([self.new_pauli().logical_x().to_bsf()])

    @property
    @functools.lru_cache()
    def logical_zs(self):
        """See :meth:`qecsim.model.StabilizerCode.logical_zs`"""
        return np.array([self.new_pauli().logical_z().to_bsf()])

    # </ StabilizerCode interface methods >

    @property
    def size(self):
        """
        Size of any side of the triangular lattice in terms of number of qubits.

        :rtype: int
        """
        return self._size

    @property
    def bound(self):
        """
        Maximum value that an index coordinate can take.

        :rtype: int
        """
        return 3 * (self.size - 1) // 2

    @classmethod
    def is_plaquette(cls, index):
        """
        Return True if the index specifies a plaquette, irrespective of lattice bounds,
        i.e. column mod 3 == 2 - (row mod 3).

        :param index: Index in the format (row, column).
        :type index: 2-tuple of int
        :return: If the index specifies a plaquette
        :rtype: bool
        """
        r, c = index
        return c % 3 == 2 - (r % 3)

    @classmethod
    def is_site(cls, index):
        """
        Return True if the index specifies a site, irrespective of lattice bounds,
        i.e. column mod 3 != 2 - (row mod 3).

        :param index: Index in the format (row, column).
        :type index: 2-tuple of int
        :return: If the index specifies a site
        :rtype: bool
        """
        return not cls.is_plaquette(index)

    def is_in_bounds(self, index):
        """
        Return True if the index is within lattice bounds inclusive, irrespective of object type.

        :param index: Index in the format (row, column).
        :type index: 2-tuple of int
        :return: If the index is within lattice bounds inclusive.
        :rtype: bool
        """
        r, c = index
        return 0 <= c <= r <= self.bound

    @property
    @functools.lru_cache()
    def _plaquette_indices(self):
        """
        Return a list of the plaquette indices of the lattice.

        Notes:

        * Each index is in the format (row, column).
        * Indices are in order of increasing column and row.

        :return: List of indices in the format (row, column).
        :rtype: list of 2-tuple of int
        """
        return [i for i in itertools.product(range(self.bound + 1), repeat=2)
                if self.is_in_bounds(i) and self.is_plaquette(i)]

    def syndrome_to_plaquette_indices(self, syndrome):
        """
        Returns the indices of the plaquettes associated with the non-commuting stabilizers identified by the syndrome.

        :param syndrome: Binary vector identifying commuting and non-commuting stabilizers by 0 and 1 respectively.
        :type syndrome: numpy.array (1d)
        :return: Two sets of plaquette indices (first set for X stabilizers, second for Z stabilizers).
        :rtype: set of 2-tuple of int, set of 2-tuple of int
        """
        x_syndrome, z_syndrome = np.hsplit(syndrome, 2)
        return (set(tuple(index) for index in np.array(self._plaquette_indices)[x_syndrome.nonzero()]),
                set(tuple(index) for index in np.array(self._plaquette_indices)[z_syndrome.nonzero()]))

    @functools.lru_cache(maxsize=2 ** 15)  # for MxN lattice, cache_size <~ 2(MN) so handle 100x100 codes.
    def virtual_plaquette_index(self, index):
        """
        For the given index of a plaquette on the lattice, returns the index of the virtual plaquette just outside the
        boundary of the same color as the plaquette.

        Notes:

        * Index is in the format (row, column).
        * Given a red plaquette, the nearest virtual plaquette will reside just outside on the red boundary (i.e. the
          boundary consisting of green and blue plaquettes).
        * The rules for other color plaquettes is found by permuting the colors in the above rule.
        * The above rules apply even if the given index is outside the boundary.

        :param index: Index identifying a plaquette in the format (row, column).
        :type index: 2-tuple of int
        :return: Index of virtual plaquette.
        :rtype: 2-tuple of int
        :raises IndexError: If index is not a plaquette index.
        """
        r, c = index
        # check valid plaquette
        if not self.is_plaquette(index):
            raise IndexError('{} is not a plaquette index.'.format(index))
        if r % 3 == 0:  # green: go to left boundary
            return r, -1
        elif r % 3 == 1:  # blue: go to lower boundary
            return self.bound + 1, c
        elif r % 3 == 2:
            return r, r + 1  # red: go to diagonal boundary

    def __eq__(self, other):
        if type(other) is type(self):
            return self._size == other._size
        return NotImplemented

    def __hash__(self):
        return hash(self._size)

    def __repr__(self):
        return '{}({!r})'.format(type(self).__name__, self.size)

    def ascii_art(self, syndrome=None, pauli=None):
        """
        Return ASCII art style lattice showing syndrome bits and Pauli operators as given.

        :param syndrome: Syndrome (optional) as binary vector.
        :type syndrome: numpy.array (1d)
        :param pauli: Color 6.6.6 Pauli (optional)
        :type pauli: Color666Pauli
        :return: ASCII art style lattice.
        :rtype: str
        """
        # See https://unicode-table.com/en/#box-drawing for box-drawing unicode characters
        x_syndrome_indices, z_syndrome_indices = set(), set()
        if syndrome is not None:
            x_syndrome_indices, z_syndrome_indices = self.syndrome_to_plaquette_indices(syndrome)
        text = []
        for r in range(self.bound + 1):
            v_text = []  # vertical / diagonal links
            h_text = []  # horizontal links + sites / stabilizers
            for c in range(r + 1):
                # vertical and diagonal links
                if r > 0:
                    if self.is_site((r, c)):
                        if c > 0:
                            if self.is_site((r - 1, c - 1)):
                                v_text.append('\\')  # \
                            else:
                                v_text.append(' ')
                        if self.is_in_bounds((r - 1, c)) and self.is_site((r - 1, c)):
                            v_text.append('|')  # |
                        else:
                            v_text.append(' ')
                    else:
                        if c > 0:
                            v_text.append(' ')
                        v_text.append(' ')
                # horizontal links + sites / stabilizers
                if self.is_site((r, c)):
                    if c > 0:
                        if self.is_site((r, c - 1)):
                            h_text.append('-')  # -
                        else:
                            h_text.append(' ')
                    if pauli:  # add pauli operator
                        op = pauli.operator((r, c))
                        h_text.append(chr(0x00B7) if op == 'I' else op)  # .
                    else:  # or dot
                        h_text.append(chr(0x00B7))  # .
                else:  # add syndrome
                    if c > 0:
                        h_text.append(' ')
                    if (r, c) in x_syndrome_indices and (r, c) in z_syndrome_indices:
                        h_text.append('Y')
                    elif (r, c) in x_syndrome_indices:
                        h_text.append('X')
                    elif (r, c) in z_syndrome_indices:
                        h_text.append('Z')
                    else:
                        h_text.append(' ')
            if r > 0:
                text.append(''.join(v_text))
            text.append(''.join(h_text))
        return '\n'.join(text)

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of color 6.6.6 Pauli for this code.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        :return: Color 6.6.6 Pauli
        :rtype: Color666Pauli
        """
        return Color666Pauli(self, bsf)
