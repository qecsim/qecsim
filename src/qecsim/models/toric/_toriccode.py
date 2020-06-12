import functools
import operator

import numpy as np

from qecsim.model import StabilizerCode, cli_description
from qecsim.models.toric import ToricPauli


@cli_description('Toric (rows INT >= 2, cols INT >= 2)')
class ToricCode(StabilizerCode):
    """
    Implements a toric code defined by its lattice size.

    In addition to the members defined in :class:`qecsim.model.StabilizerCode`, it provides several lattice methods as
    described below.

    Lattice methods:

    * Get size and shape: :meth:`size`, :meth:`shape`.
    * Find shortest translation between plaquettes: :meth:`translation`.
    * Find shortest distance between plaquettes: :meth:`distance`.
    * Resolve a syndrome to plaquettes: :meth:`syndrome_to_plaquette_indices`.
    * Construct a Pauli operator on the lattice: :meth:`new_pauli`.

    Indices:

    * Indices are in the format (lattice, row, column).
    * Qubit sites (i.e. edges) are indexed with lattice=0 and lattice=1 indicating horizontal and vertical edges,
      respectively, on the primal lattice.

    For example, site indices on a 3 x 3 toric lattice (primal lattice edges shown):
    ::

         --|----(0,0,0)----|----(0,0,1)----|----(0,0,2)--
           |               |               |
        (1,0,0)         (1,0,1)         (1,0,2)
           |               |               |
         --|----(0,1,0)----|----(0,1,1)----|----(0,1,2)--
           |               |               |
        (1,1,0)         (1,1,1)         (1,1,2)
           |               |               |
         --|----(0,2,0)----|----(0,2,1)----|----(0,2,2)--
           |               |               |
        (1,2,0)         (1,2,1)         (1,2,2)
           |               |               |

    * Stabilizer plaquettes are indexed by their northern edge with lattice=0 indicating plaquettes on the primal
      lattice, and lattice=1 indicating plaquettes on the dual lattice.
      (Equivalently, vertices are indicated by their northern edge with lattice=0 indicating vertices on the dual
      lattice, and lattice=1 indicating vertices on the primal lattice.)

    For example, plaquette indices on the primal 3 x 3 lattice (primal lattice edges shown):
    ::

         --|---------------|---------------|-------------
           |               |               |
           |    (0,0,0)    |    (0,0,1)    |    (0,0,2)
           |               |               |
         --|---------------|---------------|-------------
           |               |               |
           |    (0,1,0)    |    (0,1,1)    |    (0,1,2)
           |               |               |
         --|---------------|---------------|-------------
           |               |               |
           |    (0,2,0)    |    (0,2,1)    |    (0,2,2)
           |               |               |

    For example, plaquette indices on the dual 3 x 3 lattice (dual lattice edges shown):
    ::

        (1,2,0)    :    (1,2,1)    :    (1,2,2)    :
                   :               :               :
         - - - - - : - - - - - - - : - - - - - - - : - -
                   :               :               :
        (1,0,0)    :    (1,0,1)    :    (1,0,2)    :
                   :               :               :
         - - - - - : - - - - - - - : - - - - - - - : - -
                   :               :               :
        (1,1,0)    :    (1,1,1)    :    (1,1,2)    :
                   :               :               :
         - - - - - : - - - - - - - : - - - - - - - : - -
                   :               :               :
    """

    PRIMAL_INDEX = 0
    DUAL_INDEX = 1
    MIN_SIZE = (2, 2)

    def __init__(self, rows, columns):
        """
        Initialise new toric code.

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
                raise ValueError('ToricCode minimum size is {}.'.format(self.MIN_SIZE))
        except TypeError as ex:
            raise TypeError('ToricCode invalid parameter type') from ex
        self._size = rows, columns

    # < StabilizerCode interface methods >

    @property
    @functools.lru_cache()
    def n_k_d(self):
        """See :meth:`qecsim.model.StabilizerCode.n_k_d`"""
        rows, cols = self.size
        return 2 * rows * cols, 2, min(rows, cols)

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return 'Toric {}x{}'.format(*self.size)

    @property
    @functools.lru_cache()
    def stabilizers(self):
        """
        See :meth:`qecsim.model.StabilizerCode.stabilizers`

        Notes:

        * Includes a stabilizer for each plaquette on primal and dual. So the full set is not independent.
        """
        return np.array([self.new_pauli().plaquette(i).to_bsf() for i in self._indices])

    @property
    @functools.lru_cache()
    def logical_xs(self):
        """See :meth:`qecsim.model.StabilizerCode.logical_xs`"""
        return np.array([self.new_pauli().logical_x1().to_bsf(), self.new_pauli().logical_x2().to_bsf()])

    @property
    @functools.lru_cache()
    def logical_zs(self):
        """See :meth:`qecsim.model.StabilizerCode.logical_zs`"""
        return np.array([self.new_pauli().logical_z1().to_bsf(), self.new_pauli().logical_z2().to_bsf()])

    # </ StabilizerCode interface methods >

    @property
    def size(self):
        """
        Size of the lattice in format (rows, columns), e.g. (5, 5).

        :rtype: 2-tuple of int
        """
        return self._size

    @property
    def shape(self):
        """
        Shape of the lattice in format (lattices, rows, columns), where lattice=0 is primal and lattice=1 is dual.

        :rtype: 3-tuple of int
        """
        return tuple((2, *self.size))

    @property
    @functools.lru_cache()
    def _indices(self):
        """
        Return a list of the indices of the lattice.

        Notes:

        * Each index is in the format (lattice, row, column).
        * Indices are in order of increasing column, row and lattice.

        :return: List of indices in the format (lattice, row, column).
        :rtype: list of 3-tuple of int
        """
        return list(np.ndindex(self.shape))

    @functools.lru_cache(maxsize=2 ** 28)  # for MxN lattice, cache_size <~ 2(MN)(MN-1) so handle 100x100 codes.
    def translation(self, a_index, b_index):
        """
        Evaluate the shortest taxi-cab translation from A to B in format (row_steps, col_steps).

        Notes:

        * Indices are in the format (lattice, row, column).
        * Both indices must index the same lattice.
        * Indices are modulo the lattice shape, i.e. on a (5, 5) lattice, (2, 6, -1) indexes the same plaquette as
          (0, 1, 4).
        * Negative row_steps / col_steps indicate steps in the direction of decreasing index.

        :param a_index: Index identifying a plaquette in the format (lattice, row, column).
        :type a_index: 3-tuple of int
        :param b_index: Index identifying a plaquette in the format (lattice, row, column).
        :type b_index: 3-tuple of int
        :return: Taxi-cab translation between plaquettes.
        :rtype: 2-tuple of int
        :raises IndexError: If indices do not index the same valid lattice.
        """
        # dimensions
        shape = self.shape
        dim_lats, dim_rows, dim_cols = shape
        # indices modulo shape
        a_l, a_r, a_c = np.mod(a_index, shape)
        b_l, b_r, b_c = np.mod(b_index, shape)
        # check same lattice
        if not (a_l == b_l):
            raise IndexError('{} and {} do not index the same lattice'.format(a_index, b_index))
        # cardinal steps from A to B
        steps_north = (a_r - b_r) % dim_rows
        steps_south = (b_r - a_r) % dim_rows
        steps_west = (a_c - b_c) % dim_cols
        steps_east = (b_c - a_c) % dim_cols
        # translation steps from A to B
        row_steps = steps_south if steps_south <= steps_north else -steps_north
        col_steps = steps_east if steps_east <= steps_west else -steps_west
        return row_steps, col_steps

    @functools.lru_cache(maxsize=2 ** 28)  # for MxN lattice, cache_size <~ 2(MN)(MN-1) so handle 100x100 codes.
    def distance(self, a_index, b_index):
        """
        Evaluate the taxi-cab distance between the plaquettes indexed by A and B.

        Notes:

        * Indices are in the format (lattice, row, column).
        * Both indices must index the same lattice.
        * Indices are modulo the lattice shape, i.e. on a (5, 5) lattice, (2, 6, -1) indexes the same plaquette as
          (0, 1, 4).

        :param a_index: Index identifying a plaquette in the format (lattice, row, column).
        :type a_index: 3-tuple of int
        :param b_index: Index identifying a plaquette in the format (lattice, row, column).
        :type b_index: 3-tuple of int
        :return: Taxi-cab distance between plaquettes.
        :rtype: int
        :raises IndexError: If indices do not index the same valid lattice.
        """
        row_steps, col_steps = self.translation(a_index, b_index)
        return abs(row_steps) + abs(col_steps)

    def syndrome_to_plaquette_indices(self, syndrome):
        """
        Returns the indices of the plaquettes associated with the non-commuting stabilizers identified by the
        syndrome.

        :param syndrome: Binary vector identifying commuting and non-commuting stabilizers by 0 and 1 respectively.
        :type syndrome: numpy.array (1d)
        :return: List of plaquette indices.
        :rtype: list of 3-tuple of int
        """
        return [tuple(index) for index in np.array(self._indices)[syndrome.nonzero()]]

    def __eq__(self, other):
        if isinstance(other, ToricCode):
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
        :param pauli: Toric Pauli (optional)
        :type pauli: ToricPauli
        :return: ASCII art style lattice.
        :rtype: str
        """
        # See https://unicode-table.com/en/#box-drawing for box-drawing unicode characters
        syndrome_indices = set() if syndrome is None else self.syndrome_to_plaquette_indices(syndrome)

        def _operator_text(index):
            if pauli:
                op = pauli.operator(index)
                return chr(0x00B7) if op == 'I' else op  # .
            return chr(0x2500) if index[0] == self.PRIMAL_INDEX else chr(0x2502)  # - else |

        def _syndrome_text(index):
            if index in syndrome_indices:
                return 'Z' if index[0] == self.PRIMAL_INDEX else 'X'  # Z else X
            return ' ' if index[0] == self.PRIMAL_INDEX else chr(0x253C)  # ' ' else +

        text = []
        for row in range(self.shape[1]):
            primal_row_text, dual_row_text = [], []
            for col in range(self.shape[2]):
                primal_row_text.append(_syndrome_text((1, (row - 1) % self.shape[1], col)))
                primal_row_text.append(_operator_text((0, row, col)))
                dual_row_text.append(_operator_text((1, row, col)))
                dual_row_text.append(_syndrome_text((0, row, col)))
            text.append(chr(0x2500).join(primal_row_text))  # -.join
            text.append(' '.join(dual_row_text))  # ' '.join
        return '\n'.join(text)

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of toric Pauli for this code.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        :return: Toric Pauli
        :rtype: ToricPauli
        """
        return ToricPauli(self, bsf)
