import functools
import itertools
import operator

import numpy as np
from qecsim.model import StabilizerCode, cli_description
from qecsim.models.rotatedtoric import RotatedToricPauli


@cli_description('Rotated toric (rows INT even >= 2, cols INT even >= 2)')
class RotatedToricCode(StabilizerCode):
    r"""
    Implements a rotated toric code defined by its lattice size.

    In addition to the members defined in :class:`qecsim.model.StabilizerCode`, it provides several lattice methods as
    described below.

    Lattice methods:

    * Get size: :meth:`size`.
    * Get plaquette type: :meth:`is_x_plaquette`, :meth:`is_z_plaquette`.
    * Get and test bounds: :meth:`bounds`, :meth:`is_in_bounds`.
    * Resolve a syndrome to plaquettes: :meth:`syndrome_to_plaquette_indices`.
    * Find shortest translation between plaquettes: :meth:`translation`.
    * Construct a Pauli operator on the lattice: :meth:`new_pauli`.

    Indices:

    * Indices are in the format (x, y).
    * Qubit sites (vertices) are indexed by (x, y) coordinates with the origin at the lower left qubit.
    * Stabilizer plaquettes are indexed by (x, y) coordinates such that the lower left corner of the plaquette is on the
      qubit site at (x, y).
    * X-type stabilizer plaquette indices satisfy (x-y) % 2 == 1.
    * Z-type stabilizer plaquette indices satisfy (x-y) % 2 == 0.

    For example, qubit site indices on a 4 x 4 lattice:
    ::

          |         |         |         |
          |         |         |         |
          |         |         |         |
        (0,3)-----(1,3)-----(2,3)-----(3,3)-----
          |         |         |         |
          |         |         |         |
          |         |         |         |
        (0,2)-----(1,2)-----(2,2)-----(3,2)-----
          |         |         |         |
          |         |         |         |
          |         |         |         |
        (0,1)-----(1,1)-----(2,1)-----(3,1)-----
          |         |         |         |
          |         |         |         |
          |         |         |         |
        (0,0)-----(1,0)-----(2,0)-----(3,0)-----

    For example, stabilizer plaquette types and indices on a 4 x 4 lattice:
    ::

          |    X    |    Z    |    X    |    Z
          |  (0,3)  |  (1,3)  |  (2,3)  |  (3,3)
          |         |         |         |
          +---------+---------+---------+-------
          |    Z    |    X    |    Z    |    X
          |  (0,2)  |  (1,2)  |  (2,2)  |  (3,2)
          |         |         |         |
          +---------+---------+---------+-------
          |    X    |    Z    |    X    |    Z
          |  (0,1)  |  (1,1)  |  (2,1)  |  (3,1)
          |         |         |         |
          +---------+---------+---------+-------
          |    Z    |    X    |    Z    |    X
          |  (0,0)  |  (1,0)  |  (2,0)  |  (3,0)
          |         |         |         |
          +---------+---------+---------+-------

    """

    MIN_SIZE = (2, 2)

    def __init__(self, rows, columns):
        """
        Initialise new rotated toric code.

        :param rows: Number of rows in lattice.
        :type rows: int
        :param columns: Number of columns in lattice.
        :type columns: int
        :raises ValueError: if (rows, columns) smaller than (2, 2) in either dimension.
        :raises ValueError: if rows or columns are odd.
        :raises TypeError: if any parameter is of an invalid type.
        """
        min_rows, min_cols = self.MIN_SIZE
        try:  # paranoid checking for CLI. (operator.index ensures the parameter can be treated as an int)
            if operator.index(rows) < min_rows or operator.index(columns) < min_cols:
                raise ValueError('RotatedToricCode minimum size is {}.'.format(self.MIN_SIZE))
            if rows % 2 or columns % 2:
                raise ValueError('RotatedToricCode dimensions must be even.')
        except TypeError as ex:
            raise TypeError('RotatedToricCode invalid parameter type') from ex
        self._size = rows, columns

    # < StabilizerCode interface methods >

    @property
    @functools.lru_cache()
    def n_k_d(self):
        """See :meth:`qecsim.model.StabilizerCode.n_k_d`"""
        # n = r*c, k = 1, d = min(r, c)
        rows, cols = self.size
        return rows * cols, 2, min(rows, cols)

    @property
    def label(self):
        """See :meth:`qecsim.model.StabilizerCode.label`"""
        return 'Rotated toric {}x{}'.format(*self.size)

    @property
    @functools.lru_cache()
    def stabilizers(self):
        """See :meth:`qecsim.model.StabilizerCode.stabilizers`"""
        return np.array([self.new_pauli().plaquette(i).to_bsf() for i in self._plaquette_indices])

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
        Size of the lattice in format (rows, columns), e.g. (4, 4).

        :rtype: 2-tuple of int
        """
        return self._size

    @staticmethod
    def is_x_plaquette(index):
        """
        Return True if the plaquette index specifies an X-type plaquette, irrespective of lattice bounds.

        :param index: Index in the format (x, y).
        :type index: 2-tuple of int
        :return: If the index specifies an X-type plaquette.
        :rtype: bool
        """
        x, y = index
        return (x - y) % 2 == 1

    @staticmethod
    def is_z_plaquette(index):
        """
        Return True if the plaquette index specifies an Z-type plaquette, irrespective of lattice bounds.

        :param index: Index in the format (x, y).
        :type index: 2-tuple of int
        :return: If the index specifies an Z-type plaquette.
        :rtype: bool
        """
        x, y = index
        return (x - y) % 2 == 0

    @property
    def bounds(self):
        """
        Maximum x and y value that an index coordinate can take.

        :rtype: 2-tuple of int
        """
        # max_row, max_col
        rows, cols = self.size
        return cols - 1, rows - 1  # max_x, max_y

    def is_in_bounds(self, index):
        """
        Return True if the index is within lattice bounds inclusive.

        :param index: Index in the format (x, y).
        :type index: 2-tuple of int
        :return: If the index is within lattice bounds inclusive.
        :rtype: bool
        """
        x, y = index
        max_x, max_y = self.bounds
        return 0 <= x <= max_x and 0 <= y <= max_y

    @property
    @functools.lru_cache()
    def _plaquette_indices(self):
        """
        Return a list of the plaquette indices of the lattice.

        Notes:

        * Each index is in the format (x, y).
        * Indices are in order of increasing type, y, x. (Z-type first)

        :return: List of indices in the format (x, y).
        :rtype: list of 2-tuple of int
        """
        max_x, max_y = self.bounds
        z_plaquette_indices, x_plaquette_indices = [], []
        for y in range(max_y + 1):
            for x in range(max_x + 1):
                index = x, y
                if self.is_z_plaquette(index):
                    z_plaquette_indices.append(index)
                else:
                    x_plaquette_indices.append(index)
        return list(itertools.chain(z_plaquette_indices, x_plaquette_indices))

    def syndrome_to_plaquette_indices(self, syndrome):
        """
        Returns the indices of the plaquettes associated with the non-commuting stabilizers identified by the syndrome.

        :param syndrome: Binary vector identifying commuting and non-commuting stabilizers by 0 and 1 respectively.
        :type syndrome: numpy.array (1d)
        :return: List of plaquette indices.
        :rtype: list of 2-tuple of int
        """
        return set(tuple(index) for index in np.array(self._plaquette_indices)[syndrome.nonzero()])

    def translation(self, a_index, b_index):
        """
        Evaluate the shortest taxi-cab translation from plaquette A to plaquette B in format (x_steps, y_steps).

        Notes:

        * Indices are in the format (x, y).
        * Indices are modulo lattice dimensions, i.e. on a (2, 2) lattice, (2, -1) indexes the same site as (0, 1).
        * Both plaquettes must be of the same type, i.e. X or Z.
        * Negative x_steps / y_steps indicate steps in the direction of decreasing index.

        :param a_index: Plaquette index as (x, y).
        :type a_index: (int, int)
        :param b_index: Plaquette index as (x, y).
        :type b_index: (int, int)
        :return: Taxi-cab translation between plaquettes.
        :rtype: 2-tuple of int
        :raises IndexError: If plaquettes are not of the same type (i.e. X or Z).
        """
        # check both plaquettes are the same type
        if self.is_z_plaquette(a_index) != self.is_z_plaquette(b_index):
            raise IndexError('Path undefined between plaquettes of different types: {}, {}.'.format(a_index, b_index))
        # dimensions
        dim_y, dim_x = self.size
        # indices modulo dimensions
        a_x, a_y = np.mod(a_index, (dim_x, dim_y))
        b_x, b_y = np.mod(b_index, (dim_x, dim_y))
        # cardinal steps from A to B
        steps_north = (b_y - a_y) % dim_y
        steps_south = (a_y - b_y) % dim_y
        steps_east = (b_x - a_x) % dim_x
        steps_west = (a_x - b_x) % dim_x
        # translation steps from A to B
        x_steps = steps_east if steps_east <= steps_west else -steps_west
        y_steps = steps_north if steps_north <= steps_south else -steps_south
        return x_steps, y_steps

    def __eq__(self, other):
        if isinstance(other, RotatedToricCode):
            return self._size == other._size
        return NotImplemented

    def __hash__(self):
        return hash(self._size)

    def __repr__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, *self.size)

    def ascii_art(self, syndrome=None, pauli=None, plaquette_labels=None, site_labels=None):
        """
        Return ASCII art style lattice showing primal lattice lines with syndrome bits and Pauli operators as given.

        Notes:

        * Optional plaquette_labels override syndrome. (Out of bound indices are ignored.)
        * Optional site_labels override pauli. (Out of bound indices are ignored.)

        :param syndrome: Syndrome (optional) as binary vector.
        :type syndrome: numpy.array (1d)
        :param pauli: Rotated toric Pauli (optional)
        :type pauli: RotatedToricPauli
        :param plaquette_labels: Dictionary of plaquette indices as (x, y) to single-character labels (optional).
        :type plaquette_labels: dict of (int, int) to char
        :param site_labels: Dictionary of site indices as (x, y) to single-character labels (optional).
        :type site_labels: dict of (int, int) to char
        :return: ASCII art style lattice.
        :rtype: str
        """
        # See https://unicode-table.com/en/blocks/box-drawing/ for box-drawing unicode characters
        max_x, max_y = self.bounds
        syndrome_indices = set() if syndrome is None else self.syndrome_to_plaquette_indices(syndrome)
        pauli = self.new_pauli() if pauli is None else pauli
        plaquette_labels = {} if plaquette_labels is None else plaquette_labels
        site_labels = {} if site_labels is None else site_labels

        # Build row templates
        # e.g. (where @=plaquette, o=site):
        #
        # |#@#| @ |#@#| @   :plaquette_row_odd
        # o---o---o---o---  :site_row
        # | @ |#@#| @ |#@#  :plaquette_row_even
        # o---o---o---o---  :site_row
        # |#@#| @ |#@#| @   :plaquette_row_odd
        # o---o---o---o---  :site_row
        # | @ |#@#| @ |#@#  :plaquette_row_even
        # o---o---o---o---  :site_row
        #
        # Common chars
        c_dot = chr(0x00B7)
        c_dash = chr(0x2500)
        c_bar = chr(0x2502)
        c_shade = chr(0x2591)
        # Common char sequences
        cs_px = c_bar + c_shade + '{}' + c_shade  # '|#{}#'
        cs_pz = c_bar + ' {} '  # '| {} '
        cs_s = '{}' + c_dash * 3  # '{}---'
        # |#@#| @ |#@#| @
        t_plaquette_row_odd = ''.join(([cs_px, cs_pz] * (max_x + 1))[:max_x + 1])
        # o---o---o---o---
        t_site_row = cs_s * (max_x + 1)
        # | @ |#@#| @ |#@#
        t_plaquette_row_even = ''.join(([cs_pz, cs_px] * (max_x + 1))[:max_x + 1])

        # Parameter extraction functions
        def _site_parameters(y):
            indices = [i for i in ((x, y) for x in range(max_x + 1))]
            parameters = []
            for i in indices:
                if i in site_labels:
                    parameters.append(site_labels[i])
                else:
                    op = pauli.operator(i)
                    parameters.append(c_dot if op == 'I' else op)
            return parameters

        def _plaquette_parameters(y):
            indices = [i for i in ((x, y) for x in range(0, max_x + 1))]
            parameters = []
            for i in indices:
                is_z_plaquette = self.is_z_plaquette(i)
                if i in plaquette_labels:
                    parameters.append(plaquette_labels[i])
                elif i in syndrome_indices:
                    parameters.append('Z' if is_z_plaquette else 'X')
                else:
                    parameters.append(' ' if is_z_plaquette else c_shade)
            return parameters

        # Append templates to text with parameters
        text = []
        for y in range(max_y, -1, -1):
            if y % 2 == 0:
                text.append(t_plaquette_row_even.format(*_plaquette_parameters(y)))
            else:
                text.append(t_plaquette_row_odd.format(*_plaquette_parameters(y)))
            text.append(t_site_row.format(*_site_parameters(y)))

        return '\n'.join(text)

    def new_pauli(self, bsf=None):
        """
        Convenience constructor of rotated toric Pauli for this code.

        Notes:

        * For performance reasons, the new Pauli is a view of the given bsf. Modifying one will modify the other.

        :param bsf: Binary symplectic representation of Pauli. (Optional. Defaults to identity.)
        :type bsf: numpy.array (1d)
        :return: Rotated toric Pauli
        :rtype: RotatedToricPauli
        """
        return RotatedToricPauli(self, bsf)
