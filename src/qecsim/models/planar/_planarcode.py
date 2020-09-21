import functools
import itertools
import operator

import numpy as np

from qecsim.model import StabilizerCode, cli_description
from qecsim.models.planar import PlanarPauli


@cli_description('Planar (rows INT >= 2, cols INT >= 2)')
class PlanarCode(StabilizerCode):
    """
    Implements a planar mixed boundary code defined by its lattice size.

    In addition to the members defined in :class:`qecsim.model.StabilizerCode`, it provides several lattice methods as
    described below.

    Lattice methods:

    * Get size: :meth:`size`.
    * Find shortest translation between plaquettes: :meth:`translation`.
    * Find shortest distance between plaquettes: :meth:`distance`.
    * Resolve a syndrome to plaquettes: :meth:`syndrome_to_plaquette_indices`.
    * Construct a Pauli operator on the lattice: :meth:`new_pauli`.

    Indices:

    * Indices are in the format (row, column).
    * Qubit site (i.e. edge) indices satisfy (row + column) mod 2 = 0. On the primal lattice, horizontal edge indices
      satisfy row mod 2 = 0 and col mod 2 = 0, while vertical edge indices satisfy row mod 2 = 1 and col mod 2 = 1.

    For example, site indices on a 3 x 3 planar lattice (primal lattice edges shown):
    ::

        (0,0)-----|-----(0,2)-----|-----(0,4)
                  |               |
                (1,1)           (1,3)
                  |               |
        (2,0)-----|-----(2,2)-----|-----(2,4)
                  |               |
                (3,1)           (3,3)
                  |               |
        (4,0)-----|-----(4,2)-----|-----(4,4)

    * Stabilizer plaquette indices satisfy (row + column) mod 2 = 1. On the primal lattice, plaquette indices satisfy
      row mod 2 = 1 and col mod 2 = 0, while, on the dual lattice, plaquette indices satisfy row mod 2 = 0 and
      col mod 2 = 1.

    For example, plaquette indices on the primal 3 x 3 lattice (primal lattice edges shown):
    ::

           -------|---------------|-------
                  |               |
        (1,0)     |     (1,2)     |     (1,4)
                  |               |
           -------|---------------|-------
                  |               |
        (3,0)     |     (3,2)     |     (3,4)
                  |               |
           -------|---------------|-------

    For example, plaquette indices on the dual 3 x 3 lattice (dual lattice edges shown):
    ::

          :     (0,1)     :     (0,3)     :
          :               :               :
          : - - - - - - - : - - - - - - - :
          :               :               :
          :     (2,1)     :     (2,3)     :
          :               :               :
          : - - - - - - - : - - - - - - - :
          :               :               :
          :     (4,1)     :     (4,3)     :
    """

    MIN_SIZE = (2, 2)

    def __init__(self, rows, columns):
        """
        Initialise new planar code.

        :param rows: Number of rows in lattice.
        :type rows: int
        :param columns: Number of columns in lattice.
        :type columns: int
        :raises ValueError: if (rows, columns) smaller than (2, 2) in either dimension.
        :raises TypeError: if any parameter is of an invalid type.
        """
        min_rows, min_cols = self.MIN_SIZE
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if operator.index(rows) < min_rows or operator.index(columns) < min_cols:
                raise ValueError('PlanarCode minimum size is {}.'.format(self.MIN_SIZE))
        except TypeError as ex:
            raise TypeError('PlanarCode invalid parameter type') from ex
        self._size = rows, columns

    # < StabilizerCode interface methods >

    @property
    @functools.lru_cache()
    def n_k_d(self):
        """See :meth:`qecsim.model.StabilizerCode.n_k_d`"""
        # n = r*c horizontal edges + (r-1)*(c-1) vertical edges, k = 1, d = min(r, c)
        rows, cols = self.size
        return rows * cols + (rows - 1) * (cols - 1), 1, min(rows, cols)

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return 'Planar {}x{}'.format(*self.size)

    @property
    @functools.lru_cache()
    def stabilizers(self):
        """See :meth:`qecsim.model.StabilizerCode.stabilizers`"""
        return np.array([self.new_pauli().plaquette(i).to_bsf() for i in self._plaquette_indices])

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
        Size of the lattice in format (rows, columns), e.g. (5, 5).

        :rtype: 2-tuple of int
        """
        return self._size

    @staticmethod
    def is_plaquette(index):
        """
        Return True if the index specifies a plaquette, irrespective of lattice bounds, i.e. (row + column) mod 2 = 1.

        :param index: Index in the format (row, column).
        :type index: 2-tuple of int
        :return: If the index specifies a plaquette
        :rtype: bool
        """
        r, c = index
        return (r + c) % 2 == 1

    @staticmethod
    def is_site(index):
        """
        Return True if the index specifies a site (i.e. (row + column) mod 2 = 0), irrespective of lattice bounds.

        :param index: Index in the format (row, column).
        :type index: 2-tuple of int
        :return: If the index specifies a site.
        :rtype: bool
        """
        return not PlanarCode.is_plaquette(index)

    @staticmethod
    def is_primal(index):
        """
        Return True if the index specifies a primal plaquette (i.e. row mod 2 = 1) or site (i.e. row mod 2 = 0),
        irrespective of lattice bounds.

        :param index: Index in the format (row, column).
        :type index: 2-tuple of int
        :return: If the index specifies a primal plaquette or site.
        :rtype: bool
        """
        r, c = index
        return (PlanarCode.is_plaquette(index) and r % 2 == 1) or (PlanarCode.is_site(index) and r % 2 == 0)

    @staticmethod
    def is_dual(index):
        """
        Return True if the index specifies a dual plaquette (i.e. row mod 2 = 0) or site (i.e. row mod 2 = 1),
        irrespective of lattice bounds.

        :param index: Index in the format (row, column).
        :type index: 2-tuple of int
        :return: If the index specifies a dual plaquette or site.
        :rtype: bool
        """
        return not PlanarCode.is_primal(index)

    @property
    def bounds(self):
        """
        Maximum row and column value that an index coordinate can take.

        :rtype: 2-tuple of int
        """
        # max_row, max_col
        rows, cols = self.size
        return 2 * rows - 2, 2 * cols - 2

    def is_in_bounds(self, index):
        """
        Return True if the index is within lattice bounds inclusive, irrespective of object type.

        :param index: Index in the format (row, column).
        :type index: 2-tuple of int
        :return: If the index is within lattice bounds inclusive.
        :rtype: bool
        """
        r, c = index
        max_row, max_col = self.bounds
        return 0 <= r <= max_row and 0 <= c <= max_col

    @property
    @functools.lru_cache()
    def _plaquette_indices(self):
        """
        Return a list of the plaquette indices of the lattice.

        Notes:

        * Each index is in the format (row, column).
        * Indices are in order of increasing lattice, row and column.

        :return: List of indices in the format (row, column).
        :rtype: list of 2-tuple of int
        """
        max_row, max_col = self.bounds
        primal_plaquette_indices, dual_plaquette_indices = [], []
        for index in np.ndindex((max_row + 1, max_col + 1)):
            if self.is_plaquette(index):
                if self.is_primal(index):
                    primal_plaquette_indices.append(index)
                else:
                    dual_plaquette_indices.append(index)
        return list(itertools.chain(primal_plaquette_indices, dual_plaquette_indices))

    @functools.lru_cache(maxsize=2 ** 28)  # for MxN lattice, cache_size <~ 2(MN)(MN-1) so handle 100x100 codes.
    def translation(self, a_index, b_index):
        """
        Evaluate the shortest taxi-cab translation from plaquette A to plaquette B in format (row_steps, col_steps),
        where translation is the number of plaquette steps not the the difference in indices.

        Notes:

        * Indices are in the format (row, column).
        * Both indices must index the same lattice, see :meth:`is_primal` / :meth:`is_dual`.
        * Plaquettes not indexed within the lattice are said to be virtual, see :meth:`bounds`.
        * If both plaquettes are virtual then the translation is defined to be (0, 0).
        * Negative row_steps / col_steps indicate steps in the direction of decreasing index.

        :param a_index: Index identifying a plaquette in the format (row, column).
        :type a_index: 2-tuple of int
        :param b_index: Index identifying a plaquette in the format (row, column).
        :type b_index: 2-tuple of int
        :return: Taxi-cab translation between plaquettes.
        :rtype: 2-tuple of int
        :raises IndexError: If indices are not plaquette indices on the same lattice.
        """
        a_r, a_c = a_index
        b_r, b_c = b_index
        # check valid plaquette and same lattice indices
        if not self.is_plaquette(a_index):
            raise IndexError('{} is not a plaquette index.'.format(a_index))
        if not self.is_plaquette(b_index):
            raise IndexError('{} is not a plaquette index.'.format(b_index))
        if not self.is_primal(a_index) == self.is_primal(b_index):
            raise IndexError('{} and {} do not lie on the same lattice'.format(a_index, b_index))

        # calculate translation
        if not self.is_in_bounds(a_index) and not self.is_in_bounds(b_index):
            row_steps = 0
            col_steps = 0
        else:
            row_steps = (b_r - a_r) // 2
            col_steps = (b_c - a_c) // 2
        return row_steps, col_steps

    @functools.lru_cache(maxsize=2 ** 28)  # for MxN lattice, cache_size <~ 2(MN)(MN-1) so handle 100x100 codes.
    def distance(self, a_index, b_index):
        """
        Evaluate the taxi-cab distance between the plaquettes indexed by A and B, where distance is the number of
        plaquette steps not the difference in indices.

        Notes:

        * Indices are in the format (row, column).
        * Both indices must index the same lattice, see :meth:`is_primal` / :meth:`is_dual`.
        * Plaquettes not indexed within the lattice are said to be virtual, see :meth:`bounds`.
        * If both plaquettes are virtual then the distance is defined to be 0.

        :param a_index: Index identifying a plaquette in the format (row, column).
        :type a_index: 2-tuple of int
        :param b_index: Index identifying a plaquette in the format (row, column).
        :type b_index: 2-tuple of int
        :return: Taxi-cab distance between plaquettes.
        :rtype: int
        :raises IndexError: If indices are not plaquette indices on the same lattice.
        """
        row_steps, col_steps = self.translation(a_index, b_index)
        return abs(row_steps) + abs(col_steps)

    def syndrome_to_plaquette_indices(self, syndrome):
        """
        Returns the indices of the plaquettes associated with the non-commuting stabilizers identified by the syndrome.

        :param syndrome: Binary vector identifying commuting and non-commuting stabilizers by 0 and 1 respectively.
        :type syndrome: numpy.array (1d)
        :return: List of plaquette indices.
        :rtype: list of 2-tuple of int
        """
        return set(tuple(index) for index in np.array(self._plaquette_indices)[syndrome.nonzero()])

    @functools.lru_cache(maxsize=2 ** 15)  # for MxN lattice, cache_size <~ 2(MN) so handle 100x100 codes.
    def virtual_plaquette_index(self, index):
        """
        For the given index of a plaquette on the primal (dual) lattice, returns the index of the virtual plaquette just
        outside the nearest primal (dual) boundary.

        Notes:

        * Index is in the format (row, column).
        * Given a primal (dual) plaquette, the nearest virtual plaquette will reside on the North or South (West or
          East) boundary, so the returned index will be the same as the given index with the row (column) adjusted to
          sit just outside the nearest boundary. If both boundaries are equally close then the North (West) boundary is
          preferred.
        * The above rule applies even if the given index is outside the boundary.

        :param index: Index identifying a plaquette in the format (row, column).
        :type index: 2-tuple of int
        :return: Index of nearest virtual plaquette.
        :rtype: 2-tuple of int
        :raises IndexError: If index is not a plaquette index.
        """
        r, c = index
        # check valid plaquette
        if not self.is_plaquette(index):
            raise IndexError('{} is not a plaquette index.'.format(index))
        rows, cols = self.size
        if self.is_primal(index):
            # primal bounds
            p_min_r, p_max_r = 1, 2 * rows - 3
            if abs(r - p_min_r) <= abs(p_max_r - r):
                return p_min_r - 2, c
            else:
                return p_max_r + 2, c
        else:  # dual
            # dual bounds
            d_min_c, d_max_c = 1, 2 * cols - 3
            if abs(c - d_min_c) <= abs(d_max_c - c):
                return r, d_min_c - 2
            else:
                return r, d_max_c + 2

    def __eq__(self, other):
        if type(other) is type(self):
            return self._size == other._size
        return NotImplemented

    def __hash__(self):
        return hash(self._size)

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, *self.size)

    def ascii_art(self, syndrome=None, pauli=None):
        """
        Return ASCII art style lattice showing primal lattice lines with syndrome bits and Pauli operators as given.

        :param syndrome: Syndrome (optional) as binary vector.
        :type syndrome: numpy.array (1d)
        :param pauli: Planar Pauli (optional)
        :type pauli: PlanarPauli
        :return: ASCII art style lattice.
        :rtype: str
        """
        # See https://unicode-table.com/en/#box-drawing for box-drawing unicode characters
        max_row, max_col = self.bounds
        syndrome_indices = set() if syndrome is None else self.syndrome_to_plaquette_indices(syndrome)
        text = []
        for row in range(max_row + 1):
            row_text = []
            for col in range(max_col + 1):
                index = row, col
                if self.is_site(index):
                    if pauli:  # add pauli operator
                        op = pauli.operator(index)
                        op_text = chr(0x00B7) if op == 'I' else op  # .
                        row_text.append(op_text)
                    else:  # add grid lines
                        grid_text = chr(0x2500) if self.is_primal(index) else chr(0x2502)  # - else |
                        row_text.append(grid_text)
                    link_text = chr(0x2500) if self.is_primal(index) else ' '  # - else ' '
                else:
                    if index in syndrome_indices:  # add syndrome
                        syndrome_text = 'Z' if self.is_primal(index) else 'X'
                        row_text.append(syndrome_text)
                    else:  # add grid lines
                        if self.is_primal(index):
                            row_text.append(' ')
                        else:
                            if row == 0:
                                vertex_text = chr(0x252C)  # T
                            elif row == max_row:
                                vertex_text = chr(0x2534)  # inverted T
                            else:
                                vertex_text = chr(0x253C)  # +
                            row_text.append(vertex_text)
            text.append(link_text.join(row_text))
        return '\n'.join(text)

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of planar Pauli for this code.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        :return: Planar Pauli
        :rtype: PlanarPauli
        """
        return PlanarPauli(self, bsf)
